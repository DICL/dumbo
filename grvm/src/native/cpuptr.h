#ifndef  _CPUPTR_H_
#define  _CPUPTR_H_

#include "rpc.h"

extern "C" {

#define POFFSET_BITS PAGE_BIT

#define INVALID_BIT_MASK (((uintptr_t)1) << 63)
#define PNUM_BIT_MASK ((uintptr_t) 0x7FFFFFFFFFFFF000L)
#define POFFSET_BIT_MASK ((uintptr_t)0x0000000000000FFFL)
#define CPT_IS_INVALID(p) ((int)((((uintptr_t)p) & INVALID_BIT_MASK) >> 63))
#define CPT_SET_INVALID(p) ((cpupointer)(((uintptr_t)p) | INVALID_BIT_MASK))
#define CPT_CLR_INVALID(p) ((cpupointer)(((uintptr_t)p) & ~INVALID_BIT_MASK))
#define CPT_OFFSET(p) (((uintptr_t)p) & POFFSET_BIT_MASK)
#define CPT_PAGE_ADDRESS(p)  ((void *)((uintptr_t)p & PNUM_BIT_MASK))

typedef void *cpupointer;
extern __device__ cpupointer cpuptr_init(void *a);
extern __device__ cpupointer cpuptr_link(cpupointer p);
extern __device__ cpupointer cpuptr_unlink(cpupointer p);
extern __device__ cpupointer cpuptr_read_int(cpupointer p, int *v);
extern __device__ cpupointer cpuptr_write_int(cpupointer p, int v);
extern __device__ cpupointer cpuptr_inc(cpupointer p, int delta);
extern __device__ void cpuptr_release(cpupointer p);

__device__ inline cpupointer cpuptr_read_int_fast(cpupointer p, int *value) {
  if (__all(!CPT_IS_INVALID(p))) {
    *value = *(int *)p;
    return p;
  } else {
    return cpuptr_read_int(p, value);
  }
}

__device__ inline cpupointer cpuptr_inc_fast(cpupointer p, int offset) {
  bool intrapage;

  intrapage = offset >= 0 && (CPT_OFFSET(p) + offset) < PAGE_SIZE;
  if (__all(intrapage)) {
    p = (char *)p + offset;
    return p;
  } else {
    return cpuptr_inc(p, offset);
  }
}

}
#endif
