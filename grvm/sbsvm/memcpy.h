#ifndef MEMCPY_H
#define MEMCPY_H

#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cassert>


#ifndef PAGE_BIT
#define PAGE_BIT (12u)
//#define PAGE_BIT (21u)
//#define PAGE_BIT (14u)
#endif

#define TID (threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y)
#define LANE_ID (TID & 0x1f)
#define PAGE_SIZE (1u << PAGE_BIT)

#include "cpupointer.h"

namespace mem_cpy {
  struct Message
  {
    void *address[32];
    unsigned int size;
    volatile unsigned int state;
  };
 
  enum operation {NONE, READ, WRITE, ATOMIC_ADD};
  enum types {UNSUPPORTED, INT, UINT, CHAR, SCHAR, UCHAR, SHORT, USHORT, LONG, ULONG, LLONG, ULLONG, FLOAT, DOUBLE, LDOUBLE};
  
  extern Message *h_msg;
  extern std::size_t h_size_queue;
  extern std::uint8_t (*h_data)[PAGE_SIZE];
  extern std::uint8_t *h_out;
  extern std::uint8_t *h_in;
  extern int *h_locks;
  extern unsigned int h_request_i;
  extern cudaEvent_t end;
  extern cudaStream_t stream;

  extern __constant__ Message *d_msg;
  extern __constant__ std::size_t d_size_queue;
  extern __constant__ std::uint8_t (*d_data)[PAGE_SIZE];
  extern __constant__ int *d_locks;
  extern __device__ unsigned int d_request_i;

  __host__ void initialize(size_t size_queue = 0);
  __host__ void resize_queue(size_t size_queue);
  __host__ void finalize();
  __host__ void run_handler(void);

  __device__ void memcpy_cpu_to_gpu(void *gpu_addr, const void *cpu_addr, const unsigned int size);
  __device__ void memcpy_gpu_to_cpu(void *cpu_addr, const void *gpu_addr, const unsigned int size);


  template<typename T>
  __device__ inline unsigned int getTypeValue() {
    assert(!"unsupported type!");
    return UNSUPPORTED;
  }

  template<>
  __device__ inline unsigned int getTypeValue<int>(){
   return INT;
  }

  template<>
  __device__ inline unsigned int getTypeValue<float>(){
   return FLOAT;
  }
  
  template<>
  __device__ inline unsigned int getTypeValue<double>(){
   return DOUBLE;
  }

  template <typename T>
  __device__ void atomicAdd_gpu_to_cpu(void *cpu_addr, const T *gpu_addr, const unsigned int size)
  {
    const unsigned int mask = __ballot(1);
    const int leader = __ffs(mask) - 1;
    const int lane_id = LANE_ID;
    const int lane = __popc(mask & ((1 << lane_id) - 1));

    int i;
    if (lane_id == leader) {
      i = atomicInc(&d_request_i, d_size_queue-1);
      while (atomicExch(&d_locks[i], 1) != 0);
    }
    // locked
  
    i = __shfl(i, leader);
    Message* const request = &d_msg[i];
  
    request->address[lane] = cpu_addr;

    if (size != PAGE_SIZE) {
      std::memcpy(d_data[i*32]+(size*lane), gpu_addr, size);
    }

    if (lane_id == leader) {
      request->size = size;
      __threadfence_system();
      request->state = ATOMIC_ADD + (getTypeValue<T>() << 16);
    }

    __threadfence_system();
    while (request->state != NONE);

    if (lane_id == leader) {
      atomicExch(&d_locks[i], 0);
    }
    // unlocked
  }

}
#endif // MEMCPY_H
