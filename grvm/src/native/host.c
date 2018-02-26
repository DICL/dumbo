#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda.h>

#include "rpc.h"
#include "sbsvm.h"
#include "device.ptx.h"

#define DEBUG (0)
#define CHKCU(val)  ({                                        \
  CUresult v = (val);                                         \
  if (v != CUDA_SUCCESS) {                                    \
      const char *ename;                                      \
      cuGetErrorName(v,&ename);                               \
      fprintf(stderr, "CUDA error %s(%d) at %s:%d\n",         \
              ename, v, __FILE__, __LINE__);                  \
      exit(EXIT_FAILURE);                                     \
  }                                                           \
  v;})

sbsvmcontext sbsvm_open(size_t size_queue) {
  CUdevice device;
  size_t deviceTotalMem;
  sbsvmcontext cxt;
  int num_device_pages;

  CHKCU(cuInit(0));
  CHKCU(cuDeviceGet(&device, 0));
  CHKCU(cuDeviceTotalMem(&deviceTotalMem, device));
  cxt = (sbsvmcontext)malloc(sizeof(*cxt));
  memset(cxt, 0, sizeof (*cxt));
  CHKCU(cuCtxCreate(&cxt->cucxt, 0, device));

  if (size_queue == 0) {
    cxt->rpc.size_queue = 2 * 4096 / sizeof(struct Message);
  } else {
    cxt->rpc.size_queue = size_queue;
  }
  cxt->rpc.next_msg_index = 0;
  CHKCU(cuMemHostAlloc((void **)&cxt->rpc.messages,
      sizeof (struct Message) * cxt->rpc.size_queue,
      CU_MEMHOSTALLOC_PORTABLE));
  memset(cxt->rpc.messages, 0, sizeof (struct Message) * cxt->rpc.size_queue);
  CHKCU(cuMemAlloc((CUdeviceptr*)&cxt->rpc.locks, sizeof(int) * cxt->rpc.size_queue));
  CHKCU(cuMemsetD8((CUdeviceptr)cxt->rpc.locks, 0, sizeof(int) * cxt->rpc.size_queue));

  num_device_pages = deviceTotalMem >> (PAGE_BIT + 1);
  cxt->pcache.size = num_device_pages;
  CHKCU(cuMemAlloc((CUdeviceptr *)&cxt->pcache.tag, sizeof(struct Tag) * num_device_pages));
  CHKCU(cuMemsetD8((CUdeviceptr)cxt->pcache.tag, 0, sizeof(struct Tag) * num_device_pages));
  CHKCU(cuMemAlloc((CUdeviceptr *)&cxt->pcache.pages, PAGE_SIZE * num_device_pages));

  CHKCU(cuEventCreate(&cxt->end, CU_EVENT_DISABLE_TIMING));
  CHKCU(cuStreamCreate(&cxt->stream, CU_STREAM_NON_BLOCKING));

  if (DEBUG) {
    printf("SBSVM: allocating %d pages (%d bytes) on the device\n",
        num_device_pages, num_device_pages * PAGE_SIZE);
  }
  return cxt;
}

void sbsvm_close(sbsvmcontext cxt) {

  CHKCU(cuStreamDestroy(cxt->stream));
  CHKCU(cuEventDestroy(cxt->end));

  cxt->rpc.size_queue = 0;
  CHKCU(cuMemFreeHost(cxt->rpc.messages));
  cxt->rpc.messages = NULL;
  CHKCU(cuMemFree((CUdeviceptr)cxt->rpc.locks));
  cxt->rpc.locks = NULL;
  cxt->pcache.size = 0;
  CHKCU(cuMemFree((CUdeviceptr)cxt->pcache.pages));
  cxt->pcache.pages = NULL;
  CHKCU(cuMemFree((CUdeviceptr)cxt->pcache.tag));
  cxt->pcache.tag = NULL;
}

static void sbsvm_do_service(sbsvmcontext cxt) {
  struct Message *request;

  request = &cxt->rpc.messages[cxt->rpc.next_msg_index];
  CHKCU(cuEventRecord(cxt->end, NULL));
  while (CUDA_ERROR_NOT_READY == cuEventQuery(cxt->end)) {
    if (request->state == NONE) {
      continue;
    }

    switch (request->state) {
    case READ: {
      int i;
      const unsigned int size = request->size;
      assert(0 < size && size <= PAGE_SIZE);
          
      if (DEBUG) {
        printf("SBSVM: READ H0x%014lx -> D0x%014lx %5d\n",
          (uintptr_t)request->haddress,
          (uintptr_t)request->daddress,
          request->size);
      }
      CHKCU(cuMemcpyHtoDAsync(
          (CUdeviceptr)request->daddress,
          request->haddress,
          request->size, cxt->stream));
      CHKCU(cuStreamSynchronize(cxt->stream));
      break;
    }

    case WRITE: {
      int i;
      const unsigned int size = request->size;
      assert(0 < size && size <= PAGE_SIZE);

      if (DEBUG) {
        printf("SBSVM: WRITE D0x%014lx -> H0x%014lx\n",
          (uintptr_t)request->daddress,
          (uintptr_t)request->haddress);
        fflush(stdout);
      }

      CHKCU(cuMemcpyDtoHAsync(
          request->haddress,
          (CUdeviceptr)request->daddress,
          request->size, cxt->stream));
      CHKCU(cuStreamSynchronize(cxt->stream));

      break;
    }

    default:
      assert(!"invalid operation!");
      break;
    }
    request->haddress = NULL;
    request->daddress = NULL;
    request->size = 0;
    request->state = NONE;
    cxt->rpc.next_msg_index = (cxt->rpc.next_msg_index + 1) % cxt->rpc.size_queue;
    request = &cxt->rpc.messages[cxt->rpc.next_msg_index];
  }

  CHKCU(cuEventQuery(cxt->end));
  CHKCU(cuCtxSynchronize());
}

static void sbsvm_flush_pages(sbsvmcontext cxt) {
  struct Tag *tags;
  int i;
  size_t wb_size;
  size_t tag_size;

  wb_size = sizeof(char) * cxt->pcache.size;
  tag_size = sizeof(struct Tag) * cxt->pcache.size;
  tags = malloc(tag_size);
  CHKCU(cuMemcpyDtoH(tags, (CUdeviceptr)cxt->pcache.tag, tag_size));

  for (i = 0; i < cxt->pcache.size;i++) {
    struct Tag *ptag = &tags[i];
    if (!ptag->dirty || !ptag->residence) {
      continue;
    }
    void *haddress = tags[i].hpage;
    void *daddress = &cxt->pcache.pages[i];

    if (DEBUG) {
      printf("SBSVM: FLUSH page H0x%014lx D0x%014lx\n",
          (uintptr_t)haddress,
          (uintptr_t)daddress);
    }
    CHKCU(cuMemcpyDtoH(haddress, (CUdeviceptr)daddress, PAGE_SIZE));
  }
  free(tags);
}

static int validate_ptx_code(const char *ptxcode) {
  const char *fname;
  int fd, nwritten;
  char cmd_buf[1024];
  int len;
  int retPTXas;

  len = strlen(ptxcode);
  fname = "a.ptx";
  fd = open(fname, O_CREAT|O_RDWR|O_TRUNC, S_IRWXU);
  nwritten=write(fd, ptxcode, len);
  assert(nwritten == len);
  close(fd);
  sprintf(cmd_buf, "ptxas --gpu-name sm_35 -c -o /dev/null %s", fname);
  if (DEBUG) {
    printf("SBSVM: executing: %s\n", cmd_buf);
  }
  retPTXas = system(cmd_buf);
  if (DEBUG) {
    char dbuf[1024];
    sprintf(dbuf, "cat -n %s", fname);
    system(dbuf);
  }
  return retPTXas;
}

static char *build_ptx_code(const char *ptxcode, int ptxclen) {
  char *allptxcode;
  int appptxcode_len;
  int len_sbsvm_device_ptx;

  len_sbsvm_device_ptx= sizeof(sbsvm_device_ptx);
  appptxcode_len = len_sbsvm_device_ptx + ptxclen + 2;
  allptxcode = malloc(appptxcode_len);
  memcpy(allptxcode, sbsvm_device_ptx, len_sbsvm_device_ptx);
  allptxcode[len_sbsvm_device_ptx] = '\n';
  memcpy(&allptxcode[len_sbsvm_device_ptx + 1], ptxcode, ptxclen);
  allptxcode[appptxcode_len - 1] = '\0';
  return allptxcode;
}

void sbsvm_execute(
    sbsvmcontext ctx, const char *ptxcode, int ptxclen,
    int from, int to, void **kparams) {
  CUmodule module;
  CUfunction entry;
  int threadsPerBlock = 128;
  int gridDimX = (to - from) / threadsPerBlock;
  int blockDimX = threadsPerBlock;
  CUdeviceptr pd_cxt;
  char *allptxcode;
  int appptxcode_len;

  assert(from == 0);
  allptxcode = build_ptx_code(ptxcode, ptxclen);

  CHKCU(cuModuleLoadData(&module, allptxcode));
  CHKCU(cuModuleGetFunction(&entry, module, "kmain"));
  CHKCU(cuModuleGetGlobal(&pd_cxt, NULL, module, "d_cxt"));
  CHKCU(cuMemcpyHtoD(pd_cxt, ctx, sizeof (struct _sbsvmcontext)));

  if (DEBUG) {
    printf("SBSVM: launch_kernel %d-%d grid = %d blocks = %d\n",
        from, to, gridDimX, blockDimX);
  }

  CHKCU(cuLaunchKernel(entry,
      gridDimX, 1, 1,
      blockDimX, 1, 1,
      0, NULL,
      kparams, NULL));

  sbsvm_do_service(ctx);

  sbsvm_flush_pages(ctx);
  CHKCU(cuModuleUnload(module));
  free(allptxcode);
}

int sbsvm_validate_code(const char *ptxcode, int ptxlen) {
  char *allptxcode;
  int r;

  allptxcode = build_ptx_code(ptxcode, ptxlen);
  r = validate_ptx_code(allptxcode);
  free(allptxcode);
  return r;
}
