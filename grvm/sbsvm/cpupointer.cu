
#include "cpupointer.h"

#include "helper_cuda.h"

#ifdef DEBUG
__device__ unsigned long long MinorTime;
__device__ unsigned long long MajorTime;
__device__ unsigned long long ClearTime;
__device__ unsigned long long WbTime;
__device__ unsigned long long Minor;
__device__ unsigned long long Major;
__device__ unsigned long long Clear;
__device__ unsigned long long Wb;
__device__ unsigned long long Collision;
__device__ unsigned long long Critical;
__device__ unsigned long long Lock;
#endif

namespace cpu_pointer {
  Tag *h_tag;
  int h_size_cache;
  std::uint8_t (*h_cache)[PAGE_SIZE];
  std::uint8_t *h_wb;

  __constant__ Tag *d_tag;
  __constant__ int d_size_cache;
  __constant__ std::uint8_t (*d_cache)[PAGE_SIZE];
  __constant__ std::uint8_t *d_wb;
}

void cpu_pointer::initialize()
{
  mem_cpy::initialize();

  int device;
  checkCudaErrors(
    cudaGetDevice(&device)
  );

  cudaDeviceProp prop;
  checkCudaErrors(
    cudaGetDeviceProperties(&prop, device)
  );

  std::size_t size_mem = prop.totalGlobalMem;
  h_size_cache = size_mem >> (PAGE_BIT + 1);

  #ifdef DEBUG
  // h_size_cache = 1024*2;
  fprintf(stderr, "size_cache : %d (%f MB)\n",
         h_size_cache, PAGE_SIZE/1024.0*h_size_cache/1024.0);
  #endif


  checkCudaErrors(
    cudaMalloc(&h_tag, sizeof(*d_tag)*h_size_cache)
  );
  checkCudaErrors(
    cudaMemset(h_tag, 0, sizeof(*d_tag)*h_size_cache)
  );
  checkCudaErrors(
    cudaMemcpyToSymbol(d_tag, &h_tag, sizeof(d_tag))
  );

  checkCudaErrors(
    cudaMalloc(&h_cache, PAGE_SIZE*h_size_cache)
  );
  checkCudaErrors(
    cudaMemcpyToSymbol(d_cache, &h_cache, sizeof(d_cache))
  );

  checkCudaErrors(
    cudaMalloc(&h_wb, sizeof(*d_wb)*h_size_cache)
  );
  checkCudaErrors(
    cudaMemset(h_wb, 0, sizeof(*d_wb)*h_size_cache)
  );
  checkCudaErrors(
    cudaMemcpyToSymbol(d_wb, &h_wb, sizeof(d_wb))
  );

  checkCudaErrors(
    cudaMemcpyToSymbol(d_size_cache, &h_size_cache, sizeof(d_size_cache))
  );

#ifdef DEBUG
  unsigned long long zero = 0;
  checkCudaErrors(
    cudaMemcpyToSymbol(MinorTime, &zero, sizeof(MinorTime))
  );
  checkCudaErrors(
    cudaMemcpyToSymbol(MajorTime, &zero, sizeof(MajorTime))
  );
  checkCudaErrors(
    cudaMemcpyToSymbol(ClearTime, &zero, sizeof(ClearTime))
  );
  checkCudaErrors(
    cudaMemcpyToSymbol(WbTime, &zero, sizeof(WbTime))
  );
  checkCudaErrors(
    cudaMemcpyToSymbol(Minor, &zero, sizeof(Minor))
  );
  checkCudaErrors(
    cudaMemcpyToSymbol(Major, &zero, sizeof(Major))
  );
  checkCudaErrors(
    cudaMemcpyToSymbol(Clear, &zero, sizeof(Clear))
  );
  checkCudaErrors(
    cudaMemcpyToSymbol(Wb, &zero, sizeof(Wb))
  );
  checkCudaErrors(
    cudaMemcpyToSymbol(Collision, &zero, sizeof(Collision))
  );
  checkCudaErrors(
    cudaMemcpyToSymbol(Critical, &zero, sizeof(Critical))
  );
  checkCudaErrors(
    cudaMemcpyToSymbol(Lock, &zero, sizeof(Lock))
  );
#endif

}

void cpu_pointer::clear_cache()
{
  checkCudaErrors(
    cudaMemset(h_tag, 0, sizeof(*d_tag)*h_size_cache)
  );
}

void cpu_pointer::finalize()
{

#ifdef DEBUG
  unsigned long long tmp1, tmp2;
  checkCudaErrors(
    cudaMemcpyFromSymbol(&tmp1,MinorTime,sizeof(MinorTime),0)
  );
  checkCudaErrors(
    cudaMemcpyFromSymbol(&tmp2,MajorTime,sizeof(MajorTime),0)
  );
  tmp1-=tmp2;
  checkCudaErrors(
    cudaMemcpyToSymbol(MinorTime, &tmp1,sizeof(MinorTime))
  );
  PRINT_TIME(Minor)
  PRINT_TIME(Major)
  PRINT_TIME(Clear)
  PRINT_TIME(Wb)

  unsigned long long collision;
  checkCudaErrors(
    cudaMemcpyFromSymbol(&collision, Collision, sizeof(Collision),0)
  );
  fprintf(stderr, "Collision : %llu\n", collision);

  unsigned long long critical;
  checkCudaErrors(
    cudaMemcpyFromSymbol(&critical, Critical, sizeof(Critical),0)
  );
  fprintf(stderr, "Critical : %llu\n", critical);

  unsigned long long lock;
  checkCudaErrors(
    cudaMemcpyFromSymbol(&lock, Lock, sizeof(Lock),0)
  );
  fprintf(stderr, "lock access : %llu\n", lock);
#endif

  std::size_t zero = 0;
  checkCudaErrors(
    cudaMemcpyToSymbol(d_size_cache, &zero, sizeof(d_size_cache))
  );
  h_size_cache = zero;

  void *null = nullptr;

  checkCudaErrors(
    cudaMemcpyToSymbol(d_wb, &null, sizeof(d_wb))
  );
  checkCudaErrors(
    cudaFree(h_wb)
  );
  h_wb = nullptr;

  checkCudaErrors(
    cudaMemcpyToSymbol(d_cache, &null, sizeof(d_cache))
  );
  checkCudaErrors(
    cudaFree(h_cache)
  );
  h_cache = nullptr;

  checkCudaErrors(
    cudaMemcpyToSymbol(d_tag, &null, sizeof(d_tag))
  );
  checkCudaErrors(
    cudaFree(h_tag)
  );
  h_tag = nullptr;

  mem_cpy::finalize();
}

void cpu_pointer::run_handler()
{
  mem_cpy::run_handler();
  wb_handler();
}

void cpu_pointer::wb_handler()
{
  #ifdef DEBUG
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
  #endif

  std::uint8_t *wb = new std::uint8_t[h_size_cache];
  cudaMemcpy(wb, h_wb, sizeof(*d_wb)*h_size_cache, cudaMemcpyDeviceToHost);
  for (std::size_t i=0; i<h_size_cache; ++i) {
    if (wb[i] != 0) {
      std::uintptr_t page_addr;
      cudaMemcpy(&page_addr, &h_tag[i].address, sizeof(page_addr), cudaMemcpyDeviceToHost);
      cudaMemcpy((void*)page_addr, h_cache[i], PAGE_SIZE, cudaMemcpyDeviceToHost);
    }
  }
  delete[] wb;

  #ifdef DEBUG
    cudaEventRecord(stop);
    checkCudaErrors(cudaEventSynchronize(stop));
    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    fprintf(stderr, "wb_handler : %f ms\n", milliseconds);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
  #endif
}

void cpu_pointer::prefetch(void *ptr, long long int size)
{
  #ifdef DEBUG
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
  #endif
  
  std::uintptr_t page_addr = (std::uintptr_t)ptr & PAGE_MASK;
  size += (std::uintptr_t)ptr - page_addr;

  while (size > 0) {
    int tag_index = (page_addr >> PAGE_BIT) % h_size_cache;
    
    cudaMemcpy(h_cache[tag_index], (void*)page_addr, PAGE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(&h_tag[tag_index].address, (void*)&page_addr, sizeof(page_addr), cudaMemcpyHostToDevice);

    page_addr += PAGE_SIZE;
    size -= PAGE_SIZE;
  }

  #ifdef DEBUG
    cudaEventRecord(stop);
    checkCudaErrors(cudaEventSynchronize(stop));
    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
   fprintf(stderr, "prefetch : %f ms\n", milliseconds);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
  #endif
}