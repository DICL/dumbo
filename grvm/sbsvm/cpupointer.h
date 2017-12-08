#ifndef CPUPOINTER_H
#define CPUPOINTER_H

#include <cstdio>
#include <climits>

#include "memcpy.h"

#define TID (threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y)
#define LANE_ID (TID & 0x1f)
#define PAGE_SIZE (1u << PAGE_BIT)
#define PAGE_MASK (~((std::uintptr_t)PAGE_SIZE-1))

#define PRINT_TIME(timer) { unsigned long long tmp, blocks;   \
        cudaMemcpyFromSymbol(&tmp,timer##Time,sizeof(tmp),0); \
	cudaMemcpyFromSymbol(&blocks,timer,sizeof(blocks),0); \
        fprintf(stderr,"%s : (%llu) %fms x %llu\n", #timer, tmp, ((double)(tmp) / 1e6) / (double)(blocks), blocks); \
}

#define GET_TIME(timer) \
	asm volatile ("mov.u64 %0, %%globaltimer;" : "=l"(timer) :);

#define START(timer)			 \
	unsigned long long timer##Start; \
	{ \
		GET_TIME( timer##Start ); \
	}

#define START_WARP(timer) \
	unsigned long long timer##Start; \
	if( LANE_ID == 0 ) \
	{ \
		GET_TIME( timer##Start ); \
	}

#define STOP(timer)			\
	unsigned long long timer##Stop; \
	{ \
		GET_TIME( timer##Stop ); \
		::atomicAdd(&timer##Time, timer##Stop - timer##Start);	\
		::atomicAdd(&timer, 1);					\
	}

#define STOP_WARP(timer) \
	unsigned long long timer##Stop; \
	if( LANE_ID == 0 ) \
	{ \
		GET_TIME( timer##Stop ); \
		::atomicAdd(&timer##Time, timer##Stop - timer##Start);	\
		::atomicAdd(&timer, 1);					\
	}

#ifdef DEBUG
extern __device__ unsigned long long MinorTime;
extern __device__ unsigned long long MajorTime;
extern __device__ unsigned long long ClearTime;
extern __device__ unsigned long long WbTime;
extern __device__ unsigned long long Minor;
extern __device__ unsigned long long Major;
extern __device__ unsigned long long Clear;
extern __device__ unsigned long long Wb;
extern __device__ unsigned long long Collision;
extern __device__ unsigned long long Critical;
extern __device__ unsigned long long Lock;
#endif

namespace cpu_pointer {

  struct Tag {
    std::uintptr_t address;
    unsigned int count;
    int lock;
    int read_mode;
    unsigned reads;
  };

  extern Tag *h_tag;
  extern int h_size_cache;
  extern std::uint8_t (*h_cache)[PAGE_SIZE];
  extern std::uint8_t *h_wb;
  
  extern __constant__ Tag *d_tag;
  extern __constant__ int d_size_cache;
  extern __constant__ std::uint8_t (*d_cache)[PAGE_SIZE];
  extern __constant__ std::uint8_t *d_wb;

  __host__ void initialize();
  __host__ void clear_cache();
  __host__ void finalize();
  __host__ void run_handler();
  __host__ void wb_handler();
  __host__ void prefetch(void *addr, long long int size);

  template <typename T>
  __device__ inline T atomicAdd(T* address, const T val) {
    return ::atomicAdd(address, val);
  }

  #if __CUDA_ARCH__ < 600
  template <>
  __device__ inline double atomicAdd<double>(double* address, const double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
  }
  #endif

   template <typename T>
  __device__ inline T atomicSub(T* address, const T val) {
    return ::atomicSub(address, val);
  }

  #if __CUDA_ARCH__ < 600
  template <>
  __device__ inline double atomicSub<double>(double* address, const double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(__longlong_as_double(assumed) - val));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
  }
  #endif

  template <typename T>
  __device__ inline T atomicExch(T* address, const T val) {
    return ::atomicExch(address, val);
  }

  template <>
    __device__ inline std::uintptr_t atomicExch<std::uintptr_t>(std::uintptr_t* address, const std::uintptr_t val) {
    union {
      std::uintptr_t ptr;
      #if UINTPTR_MAX == ULLONG_MAX
      unsigned long long int i;
      #elif UINTPTR_MAX == UINT_MAX
      unsigned int i;
      #endif
    };
    ptr = val;
    i = ::atomicExch((decltype(&i))address, i);
    return ptr;
  }
  
}

#endif // CPUPOINTER_H
