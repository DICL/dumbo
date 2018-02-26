#include <stdint.h>
#include <assert.h>
#include "rpc.h"
#include "cpuptr.h"

extern "C" {

#define TID (threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y)
#define DEBUG (0)

__device__ static inline unsigned int lane_id() {
    unsigned ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__device__ static inline unsigned int warp_id() {
  unsigned ret;
  asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
  return ret;
}

__device__ void *shfl_address(void *v, int src) {
  union {
    void *addr;
    int ivals[2];
  } bval;

  bval.addr = v;
  bval.ivals[0] = __shfl(bval.ivals[0], src);
  bval.ivals[1] = __shfl(bval.ivals[1], src);
  return bval.addr;
}

__device__ struct _sbsvmcontext d_cxt;

__device__ static void memcpy_h2d(
    void *daddress, const void *haddress,
    const unsigned int size) {
    struct Message *request;
  int msg_idx;

  msg_idx = atomicInc(&d_cxt.rpc.next_msg_index, d_cxt.rpc.size_queue - 1);
  while (atomicExch(&d_cxt.rpc.locks[msg_idx], 1) != 0);

  if (DEBUG >= 1) {
    printf("T[%3d] memcpy_h2d: H0x%016lx D0x%016lx S = %5d idx = %3d\n",
        TID, haddress, daddress, size, msg_idx);
  }

  request = &d_cxt.rpc.messages[msg_idx];
  request->haddress = (void *)haddress;
  request->daddress = (void *)daddress;
  request->size = size;

  __threadfence_system();
  request->state = READ;

  __threadfence_system();
  while (request->state != NONE);

  atomicExch(&d_cxt.rpc.locks[msg_idx], 0);
}

 __device__ static void memcpy_d2h(
     void *haddr, const void *daddr,
     const unsigned int size){
    int msg_idx;
  Message* request;
  
  msg_idx = atomicInc(&d_cxt.rpc.next_msg_index, d_cxt.rpc.size_queue - 1);
  while (atomicExch(&d_cxt.rpc.locks[msg_idx], 1) != 0);
  request = &d_cxt.rpc.messages[msg_idx];
 
  request->haddress = haddr;
  request->daddress = (void *)daddr;
  request->size = size;

   __threadfence_system();
  request->state = WRITE;
 
  __threadfence_system();
  while (request->state != NONE);

  atomicExch(&d_cxt.rpc.locks[msg_idx], 0);
}


__device__ inline static int dptr2hindex(cpupointer p) {
  int idx;
  idx = (((uintptr_t)p - (uintptr_t)d_cxt.pcache.pages)
      & PNUM_BIT_MASK) >> PAGE_BIT;
  return idx;
}

__device__ static cpupointer unlink(cpupointer p) {
  unsigned int prefCount = 0;
  int tag_index;
  struct Tag *ptag;

  if (CPT_IS_INVALID(p)) {
    return p;
  }

  tag_index = dptr2hindex(p);
  ptag = &d_cxt.pcache.tag[tag_index];
  p = CPT_SET_INVALID(p);

	for(bool check = true;check;) {
    int i = __ffs(__ballot(check)) - 1;
    int checker = __shfl(tag_index, i);
    bool c = tag_index == checker;
    unsigned int num = __popc(__ballot(c));
    if (i == lane_id()) {
      prefCount = num;
    }
    if (c) {
      check = false;
    }
  }

	if (prefCount) {
	  atomicSub(&ptag->ref_counts, prefCount);
	}
	return p;
}

__device__ static void *handlePageFault(
    int warpLeader, void *hpage, int pageRefCount) {
  unsigned int hindex;
  struct Tag *ptag;
  bool isWarpLeader;
  void *dpage;

  isWarpLeader = warpLeader == lane_id();
  if (isWarpLeader) {
    if (DEBUG >= 3) {
      printf("T[%3d] handlePageFault H0x%016lx count = %3d\n",
          TID, hpage, pageRefCount);
    }

    hindex = (((uintptr_t)hpage & PNUM_BIT_MASK) >> POFFSET_BITS)
          % d_cxt.pcache.size;
    ptag = &d_cxt.pcache.tag[hindex];
    while (atomicExch(&ptag->lock, 1) != 0) ;
    if (ptag->residence) {
      if (ptag->hpage == hpage) {
        dpage = d_cxt.pcache.pages[hindex];
        ptag->ref_counts += pageRefCount;
        if (DEBUG >= 2) {
          printf("T[%3d] minor fault: H0x%016lx D0x%016lx hindex = %5d\n",
              TID, hpage, dpage, hindex);
        }
      } else if (ptag->ref_counts == 0) {
        dpage = d_cxt.pcache.pages[hindex];
        memcpy_d2h(ptag->hpage, dpage, PAGE_SIZE);
        memcpy_h2d(dpage, hpage, PAGE_SIZE);
        if (DEBUG >= 1) {
          printf("T[%3d] major fault: H0x%016lx D0x%016lx hindex = %5d\n",
              TID, hpage, dpage, hindex);
        }
        ptag->hpage = hpage;
        ptag->residence = 1;
      } else {
        dpage = NULL;
      }
    } else {
      dpage = d_cxt.pcache.pages[hindex];
      if (DEBUG >= 1) {
        printf("T[%3d] major fault: H0x%016lx D0x%016lx hindex = %5d\n",
            TID, hpage, dpage, hindex);
      }
      memcpy_h2d(dpage, hpage, PAGE_SIZE);
      ptag->hpage = hpage;
      ptag->ref_counts = pageRefCount;
      ptag->residence = 1;
    }
    atomicExch(&ptag->lock, 0);
  }
  dpage = shfl_address(dpage, warpLeader);
  return dpage;
}

__device__ static cpupointer doPageFault(cpupointer p) {
  bool hasWork;
  void *hpage;
  void *dpage;
  void *bhpage;
  bool shouldHandlePageFault;
  unsigned int pageRefCount;
  int warpLeader;

  if (CPT_IS_INVALID(p)) {
    hpage = CPT_PAGE_ADDRESS(p);
  } else {
    hpage = NULL;
  }

  hasWork = CPT_IS_INVALID(p) != 0;
  while(true) {
    warpLeader = __ffs(__ballot(hasWork));
    if (warpLeader == 0) {
      break;
    }
    warpLeader = warpLeader - 1;
    if (DEBUG >= 3) {
      printf("T[%3d]: doPageFault leader =%3d\n", TID, warpLeader);
    }
    bhpage = shfl_address(hpage, warpLeader);
    shouldHandlePageFault = (bhpage == hpage) && hasWork;
    pageRefCount = __popc(__ballot(shouldHandlePageFault));

    dpage = handlePageFault(warpLeader, bhpage, pageRefCount);
    if (shouldHandlePageFault) {
      hasWork = false;
      if (dpage) {
        p = (char *)((uintptr_t)dpage | CPT_OFFSET(p));
        p = CPT_CLR_INVALID(p);
      }
    }
  }
  return p;
}

__device__ cpupointer cpuptr_init(void *a) {
  return CPT_SET_INVALID(a);
}

__device__ cpupointer cpuptr_link(cpupointer p) {
  if (__any(CPT_IS_INVALID(p))) {
    p = doPageFault(p);
  }
  return p;
}

__device__ cpupointer cpuptr_unlink(cpupointer p) {
  p = CPT_SET_INVALID(p);
  return p;
}

__device__ cpupointer cpuptr_read_int(cpupointer p, int *value) {
  int val;
  void *hptr;

  if (__any(CPT_IS_INVALID(p))) {
    p = doPageFault(p);
  }
  if (!CPT_IS_INVALID(p)) {
    *value = *(int *)p;
  } else {
    printf("slow read\n");
    hptr = CPT_CLR_INVALID(p);
    memcpy_h2d(value, hptr, sizeof (val));
  }
  return p;
}

__device__ cpupointer cpuptr_write_int(cpupointer p, int value) {
  int hindex;
  void *hptr;

  if (__any(CPT_IS_INVALID(p))) {
    p = doPageFault(p);
  }

  if (!CPT_IS_INVALID(p)) {
    hindex = (((uintptr_t)p - (uintptr_t)d_cxt.pcache.pages)
            & PNUM_BIT_MASK) >> PAGE_BIT;
    *(int *)p = value;
    d_cxt.pcache.tag[hindex].dirty = 1;
    __threadfence();
  } else {
    printf("slow write\n");
    hptr = CPT_CLR_INVALID(p);
    memcpy_d2h(hptr, &value, sizeof(value));
  }
  return p;
}

__device__ cpupointer cpuptr_inc(cpupointer p, int offset) {
  struct Tag *ptag;
  int hindex;
  void *new_hpage, *old_hpage;
  char *old_haddr, *new_haddr;

  if (DEBUG) {
    printf("inc: 0x%012x offset = %4d\n", p, offset);
  }
  if (offset >= 0 && (CPT_OFFSET(p) + offset) < PAGE_SIZE) {
    p = (cpupointer) ((uintptr_t)p + offset);
  } else if (CPT_IS_INVALID(p)) {
    p = (cpupointer) ((uintptr_t)p + offset);  
  } else {
    hindex = (((uintptr_t)p - (uintptr_t)d_cxt.pcache.pages) & PNUM_BIT_MASK) >> PAGE_BIT;
    ptag = &d_cxt.pcache.tag[hindex];
    old_hpage = ptag->hpage;
    old_haddr = (char *) ((uintptr_t)old_hpage | CPT_OFFSET(p));
    new_haddr = old_haddr + offset;
    new_hpage = (void *)((uintptr_t)new_haddr & PNUM_BIT_MASK);
    if (DEBUG) {
      printf("t[%3d]p = 0x%012lx hindex = %5d old_hpage = %012lx old=%lx new=%lx\n",
          TID, p, hindex, old_hpage, old_haddr, new_haddr);
    }
    if (old_hpage == new_hpage) {
      p = (cpupointer)(((uintptr_t)p) + offset);
    } else {
      p = unlink(p);
      p = CPT_SET_INVALID(new_haddr);
    }
  }
  return p;
}

__device__ void cpuptr_release(cpupointer p) {
  if (__any(!CPT_IS_INVALID(p))) {
    unlink(p);
  }
}

}
