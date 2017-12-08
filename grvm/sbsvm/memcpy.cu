#include "memcpy.h"

#include "helper_cuda.h"

namespace mem_cpy {
  Message *h_msg;
  std::size_t h_size_queue;
  std::uint8_t (*h_data)[PAGE_SIZE];
  std::uint8_t *h_out;
  std::uint8_t *h_in;
  int *h_locks;
  unsigned int h_request_i;
  cudaEvent_t end;
  cudaStream_t stream;

  __constant__ Message *d_msg;
  __constant__ std::size_t d_size_queue;
  __constant__ std::uint8_t (*d_data)[PAGE_SIZE];
  __constant__ int *d_locks;
  __device__ unsigned int d_request_i;

  void atomicAdd(unsigned int types, void *des, void *src, unsigned int size);
}

__host__ void mem_cpy::initialize(std::size_t size_queue)
{
  if (size_queue == 0) {
    size_queue = 2*4096/sizeof(Message);
  }

  #ifdef DEBUG
  fprintf(stderr, "PAGE_SIZE : %.2fk\n", PAGE_SIZE/1024.0);
  fprintf(stderr, "sizeof(Message) : %zu\n", sizeof(Message));
  fprintf(stderr, "size_queue : %zu\n", size_queue);
  #endif

  resize_queue(size_queue);

  checkCudaErrors(
    cudaHostAlloc(&h_out, PAGE_SIZE*32, cudaHostAllocPortable | cudaHostAllocWriteCombined)
  );
  
  checkCudaErrors(
    cudaHostAlloc(&h_in, PAGE_SIZE*32, cudaHostAllocPortable)
  );

  checkCudaErrors(
    cudaEventCreateWithFlags(&end, cudaEventDisableTiming)
  );

  checkCudaErrors(
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)
  );

}

__host__ void mem_cpy::resize_queue(size_t size_queue)
{

  if (h_data != nullptr) {
    checkCudaErrors(
      cudaFree(h_data)
    );
  }

  if (h_locks != nullptr) {
    checkCudaErrors(
      cudaFree(h_locks)
    );
  }

  if (h_msg != nullptr) {
    checkCudaErrors(
      cudaFreeHost(h_msg)
    );
  }
  
  const std::size_t total_size_msg = sizeof(*d_msg)*size_queue;
  checkCudaErrors(
    cudaHostAlloc(&h_msg, total_size_msg, cudaHostAllocPortable)
  );
  std::memset(h_msg, 0, total_size_msg);
  checkCudaErrors(
    cudaMemcpyToSymbol(d_msg, &h_msg, sizeof(d_msg))
  );

  checkCudaErrors(
    cudaMalloc(&h_locks, sizeof(*d_locks)*size_queue)
  );
  checkCudaErrors(
    cudaMemset(h_locks, 0, sizeof(*d_locks)*size_queue)
  );
  checkCudaErrors(
    cudaMemcpyToSymbol(d_locks, &h_locks, sizeof(d_locks))
  );

  checkCudaErrors(
    cudaMalloc(&h_data, PAGE_SIZE*32*size_queue)
  );
  checkCudaErrors(
    cudaMemcpyToSymbol(d_data, &h_data, sizeof(d_data))
  );

  h_request_i = 0;
  checkCudaErrors(
    cudaMemcpyToSymbol(d_request_i, &h_request_i, sizeof(d_request_i))
  );

  h_size_queue = size_queue;
  checkCudaErrors(
    cudaMemcpyToSymbol(d_size_queue, &h_size_queue, sizeof(d_size_queue))
  );
}

__host__ void mem_cpy::finalize()
{
  checkCudaErrors(
    cudaStreamDestroy(stream)
  );

  checkCudaErrors(
    cudaEventDestroy(end)
  );

  checkCudaErrors(
    cudaFreeHost(h_in)
  );
  h_in = nullptr;

  checkCudaErrors(
    cudaFreeHost(h_out)
  );
  h_out = nullptr;

  void *null = nullptr;

  checkCudaErrors(
    cudaMemcpyToSymbol(d_data, &null, sizeof(d_data))
  );
  checkCudaErrors(
    cudaFree(h_data)
  );
  h_data = nullptr;

  h_size_queue = 0;
  checkCudaErrors(
    cudaMemcpyToSymbol(d_size_queue, &h_size_queue, sizeof(d_size_queue))
  );

  checkCudaErrors(
    cudaMemcpyToSymbol(d_locks, &null, sizeof(d_locks))
  );
  checkCudaErrors(
    cudaFree(h_locks)
  );
  h_locks = nullptr;

  checkCudaErrors(
    cudaMemcpyToSymbol(d_msg, &null, sizeof(d_msg))
  );
  checkCudaErrors(
    cudaFreeHost(h_msg)
  );
  h_msg = nullptr;
}

__host__ void mem_cpy::run_handler(void)
{
  checkCudaErrors(
    cudaEventRecord(end)
  );

  Message *request = &h_msg[h_request_i];

  while (cudaErrorNotReady == cudaEventQuery(end)) {
    if (request->state != 0) {
      const unsigned int size = request->size;
      assert(0 < size && size <= PAGE_SIZE);

      switch (request->state & 0xFFFF) {
      case 1: {
        int i;
        for (i=0; i<32 && request->address[i]; ++i) {
          std::memcpy(&h_out[size*i], request->address[i], size);
	  if (size == PAGE_SIZE) {
	    using namespace cpu_pointer;
	    int tag_index = ((std::uintptr_t)request->address[i] >> PAGE_BIT)
		            % h_size_cache;
	    checkCudaErrors(
              cudaMemcpyAsync(h_cache[tag_index], &h_out[PAGE_SIZE*i], PAGE_SIZE,
		              cudaMemcpyHostToDevice, stream)
	    );
	  }
          request->address[i] = nullptr;
        }

	if (size != PAGE_SIZE) {
          checkCudaErrors(
            cudaMemcpyAsync(h_data[h_request_i*32], h_out, size*i,
		                cudaMemcpyHostToDevice, stream)
          );
	}

        checkCudaErrors(
          cudaStreamSynchronize(stream)
        );
      }
      break;
      
      case 2: {
        int threads;
	for (threads=0; threads<32 && request->address[threads]; ++threads);

	if (size != PAGE_SIZE) {
          checkCudaErrors(
            cudaMemcpyAsync(h_in, h_data[h_request_i*32], size*threads,
	                    cudaMemcpyDeviceToHost, stream)
          );
	  checkCudaErrors(
            cudaStreamSynchronize(stream)
          );
	}
	
        for (int i=0; i < threads; ++i) {
          if (size == PAGE_SIZE) {
	    using namespace cpu_pointer;
	    int tag_index = ((std::uintptr_t)request->address[i] >> PAGE_BIT)
		            % h_size_cache;
	    checkCudaErrors(
              cudaMemcpyAsync(&h_in[PAGE_SIZE*i], h_cache[tag_index], PAGE_SIZE,
		              cudaMemcpyDeviceToHost, stream)
            );
	    checkCudaErrors(
              cudaStreamSynchronize(stream)
            );
	  }
	  std::memcpy(request->address[i], &h_in[size*i], size);
          request->address[i] = nullptr;
        }
      }
      break;

      case 3: {
        int threads;
	for (threads=0; threads<32 && request->address[threads]; ++threads);

	if (size != PAGE_SIZE) {
          checkCudaErrors(
            cudaMemcpyAsync(h_in, h_data[h_request_i*32], size*threads,
	                    cudaMemcpyDeviceToHost, stream)
          );
	  checkCudaErrors(
            cudaStreamSynchronize(stream)
          );
	}

        unsigned int types = request->state >> 16;

        for (int i=0; i < threads; ++i) {
          if (size == PAGE_SIZE) {
	    using namespace cpu_pointer;
	    int tag_index = ((std::uintptr_t)request->address[i] >> PAGE_BIT)
		            % h_size_cache;
	    checkCudaErrors(
              cudaMemcpyAsync(&h_in[PAGE_SIZE*i], h_cache[tag_index], PAGE_SIZE,
		              cudaMemcpyDeviceToHost, stream)
            );
	    checkCudaErrors(
              cudaStreamSynchronize(stream)
            );
	  }
	  mem_cpy::atomicAdd(types, request->address[i], &h_in[size*i], size);
          request->address[i] = nullptr;
        }
      }
      break;
      
      default:
      assert(!"invalid operation!");
      }

      request->state = 0;
      h_request_i = (h_request_i + 1) % h_size_queue;
      request = &h_msg[h_request_i];
    }
  }

  checkCudaErrors(
    cudaEventQuery(end)
  );
}

template <typename T>
inline void atomicAdd_helper(T* d, T* s, unsigned int n)
{
  for (unsigned int i=0; i<n; ++i) {
    *d++ += *s++;
  }
}

void mem_cpy::atomicAdd(unsigned int types, void *des, void *src, unsigned int size)
{
  switch (types) {
  case INT:
    atomicAdd_helper((int*)des, (int*)src, size/sizeof(int));
    break;
    
  case FLOAT:
    atomicAdd_helper((float*)des, (float*)src, size/sizeof(float));
    break;
    
  case DOUBLE:
    atomicAdd_helper((double*)des, (double*)src, size/sizeof(double));
    break;

  default:
    assert(!"unsupported type");
  }
}

__device__ void mem_cpy::memcpy_cpu_to_gpu(void *gpu_addr, const void *cpu_addr, const unsigned int size)
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

  request->address[lane] = (void*)cpu_addr;
  
  if (lane_id == leader) {
    request->size = size;
    __threadfence_system();
    request->state = READ;
  }

  __threadfence_system();
  while (request->state != NONE);

  if (size != PAGE_SIZE) {
    std::memcpy(gpu_addr, d_data[i*32]+(size*lane), size);
  }
  
  if (lane_id == leader) {
    atomicExch(&d_locks[i], 0);
  }
  // unlocked
}

__device__ void mem_cpy::memcpy_gpu_to_cpu(void *cpu_addr, const void *gpu_addr, const unsigned int size)
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
    request->state = WRITE;
  }

  __threadfence_system();
  while (request->state != NONE);

  if (lane_id == leader) {
    atomicExch(&d_locks[i], 0);
  }
  // unlocked
}
