#ifndef CPTR_CU_H
#define CPTR_CU_H

#include "cpupointer.h"

template <typename T>
class CPtr
{
public:
  __device__ CPtr(T *ptr)
    : ptr(ptr), page_addr(0), tag_index(0), tag(nullptr), buffer(nullptr), valid(false)
  {
    assert(sizeof(T) < PAGE_SIZE);
    setValues();
  }
  __device__ CPtr(const CPtr &cptr)
    : ptr(cptr.ptr), page_addr(0), tag_index(0), tag(nullptr), buffer(nullptr), valid(false)
  {
    assert(sizeof(T) < PAGE_SIZE);
    setValues();
  }
  __device__ ~CPtr()
  {
    clear();
  }

  __device__ CPtr& operator = (T *rhs)
  {
    ptr = rhs;
    valid = false;
    setValues();
    return *this;
  }

  __device__ CPtr& operator = (const CPtr &rhs)
  {
    ptr = rhs.ptr;
    valid = false;
    setValues();
    return *this;
  }
  __device__ CPtr& operator ++ ()
  {
    ++buffer;
    ++ptr;
    std::uintptr_t p = (std::uintptr_t)ptr & PAGE_MASK;
    bool invalid = page_addr != p;
    if (invalid) {
      setInvalid();
      setValues();
    }
    return *this;
  }

  __device__ CPtr& operator -- ()
  {
    --buffer;
    --ptr;
    std::uintptr_t p = (std::uintptr_t)ptr & PAGE_MASK;
    bool invalid = page_addr != p;
    if (invalid) {
      setInvalid();
      setValues();
    }
    return *this;
  }

  __device__ CPtr& operator += (const std::ptrdiff_t val) {
    buffer += val;
    ptr += val;
    std::uintptr_t p = (std::uintptr_t)ptr & PAGE_MASK;
    bool invalid = page_addr != p;
    if (invalid) {
      setInvalid();
      setValues();
    }
    return *this;
  }

  __device__ CPtr& operator -= (const std::ptrdiff_t val) {
    buffer -= val;
    ptr -= val;
    std::uintptr_t p = (std::uintptr_t)ptr & PAGE_MASK;
    bool invalid = page_addr != p;
    if (invalid) {
      setInvalid();
      setValues();
    }
    return *this;
  }


  __device__ T read();
  __device__ void write(const T &value);
  __device__ void atomicAdd(const T &value);

  __device__ void doCache();
  __device__ void clear();

private:
  __device__ void setValid();
  __device__ void setInvalid();
  __device__ void setValues();

  T *ptr;
  bool valid;

  // temp
  union {
    std::uintptr_t page_addr;
    int checkers[2];
  };
  int tag_index;
  T *buffer;
  cpu_pointer::Tag *tag;
};

template <typename T>
__device__ inline T CPtr<T>::read()
{
  doCache();
  
  T val;
  if (valid) {
    val = *buffer;
  } else {
    mem_cpy::memcpy_cpu_to_gpu(&val, ptr, sizeof(val));
  }

  return val;
}

template <typename T>
__device__ inline void CPtr<T>::write(const T &value)
{
  using namespace cpu_pointer;
  doCache();
  
  if (valid) {
    *buffer = value;
    __threadfence();
    d_wb[tag_index] = 1;  
  } else {
    int leader;
    bool check = true;
    while (check) {
      int i = __ffs(__ballot(check)) - 1;
      int index = __shfl(tag_index, i);
      if (index == tag_index) {
	leader = i;
        check = false;
      }
    }

    int unlocked = 1;
    while (unlocked) {
      if (leader == LANE_ID) {
        unlocked = cpu_pointer::atomicExch(&tag->lock, 1);
      }
      unlocked = __shfl(unlocked, leader);

      if (!unlocked) {
        // locked
        if (tag->address == page_addr) {
          *buffer = value;
	  __threadfence();
          d_wb[tag_index] = 1;
        } else {
	  mem_cpy::memcpy_gpu_to_cpu(ptr, &value, sizeof(value));
        }
        if (leader == LANE_ID) {
          cpu_pointer::atomicExch(&tag->lock, 0);
          // unlocked
        }
      }
    }
  }
}


template <typename T>
__device__ inline void CPtr<T>::atomicAdd(const T &value)
{
  using namespace cpu_pointer;
  
  doCache();

  if (valid) {
    cpu_pointer::atomicAdd(buffer, value);
    d_wb[tag_index] = 1;
  } else {
    int leader;
    bool check = true;
    while (check) {
      int i = __ffs(__ballot(check)) - 1;
      int index = __shfl(tag_index, i);
      if (index == tag_index) {
	leader = i;
        check = false;
      }
    }

    int unlocked = 1;
    while (unlocked) {
      if (leader == LANE_ID) {
        unlocked = cpu_pointer::atomicExch(&tag->lock, 1);
      }
      unlocked = __shfl(unlocked, leader);

      if (!unlocked) {
        // locked
        if (tag->address == page_addr) {
          cpu_pointer::atomicAdd(buffer, value);
          d_wb[tag_index] = 1;
        } else {
	  mem_cpy::atomicAdd_gpu_to_cpu(ptr, &value, sizeof(value));
        }
        if (leader == LANE_ID) {
          cpu_pointer::atomicExch(&tag->lock, 0);
          // unlocked
        }
      }
    }
  }
}

template <typename T>
__device__ inline void CPtr<T>::doCache()
{
  if (__any(!valid))
    setValid();
}

template <typename T>
__device__ void CPtr<T>::clear()
{
  if (__any(valid))
    setInvalid();
}


template <typename T>
__device__ inline void CPtr<T>::setValid()
{
  using namespace cpu_pointer;

  if (valid == false) {
    valid = true;

    #ifdef DEBUG
      START(Minor)
    #endif
    
    unsigned int count = 0;
    bool check = true;
    while (check) {
      int i = __ffs(__ballot(check)) - 1;
      int checker0 = __shfl(checkers[0], i);
      int checker1 = __shfl(checkers[1], i);
      bool c = checkers[0] == checker0 && checkers[1] == checker1;
      unsigned int num = __popc(__ballot(c));
      if (i == LANE_ID) {
        count = num;
      }
      if (c) {
        check = false;
      }
    }

    bool collision = false;
    if (count) {
      bool exit = false;
      while (!exit) {
	if (!tag->read_mode) {
	  #ifdef DEBUG
	    cpu_pointer::atomicAdd(&Lock, 1LLU);
	  #endif
	  if (cpu_pointer::atomicExch(&tag->lock, 1) == 0) {
	    
	    __threadfence();
	    if (!tag->read_mode) {
	      // write mode
	      
	      while (tag->reads != 0) {
		// wait
		__threadfence();
	      } 

	      __threadfence();
	      if (tag->count == 0) {
		// miss
	      
	        if (d_wb[tag_index] != 0) {
                  // wb
	          #ifdef DEBUG
	            START(Wb)
	          #endif
		      
	          d_wb[tag_index] = 0;
		  __threadfence();
	          mem_cpy::memcpy_gpu_to_cpu((void*)tag->address, d_cache[tag_index], PAGE_SIZE);
	     
	          #ifdef DEBUG
	            STOP(Wb)
	          #endif
	        }
	        
	        #ifdef DEBUG
                  START(Major)
                #endif
		      
	        mem_cpy::memcpy_cpu_to_gpu(d_cache[tag_index], (void*)page_addr, PAGE_SIZE);
		cpu_pointer::atomicExch(&tag->address, page_addr);
                cpu_pointer::atomicExch(&tag->count, count);
	      
	        #ifdef DEBUG
                  STOP(Major)
                #endif

	        exit = true;
	      }

	      cpu_pointer::atomicExch(&tag->read_mode, 1);
	    }
	    cpu_pointer::atomicExch(&tag->lock, 0);
            // unlocked
	  }
	} else {
	  // read
	  cpu_pointer::atomicAdd(&tag->reads, 1U);

	  __threadfence();
	  if (tag->read_mode) {
	  
	    if (tag->address == page_addr) {
	      // hit
              cpu_pointer::atomicAdd(&tag->count, count);
	      exit = true;
	    } else {
	      // hash collision
	      #ifdef DEBUG
	        cpu_pointer::atomicAdd(&Collision, 1LLU);
	      #endif

	      if (tag->count == 0) {
	        cpu_pointer::atomicExch(&tag->read_mode, 0);
	      } else {
	        collision = true;
	        #ifdef DEBUG
	          cpu_pointer::atomicAdd(&Critical, 1LLU);
	        #endif
	        exit = true;
	      }
	   
	    }
	  }
	  cpu_pointer::atomicSub(&tag->reads, 1U);
        }
      }
    }

    bool handled = !collision;
    while (any(!handled)) {
      int i = __ffs(__ballot(!handled)) - 1;
      int checker0 = __shfl(checkers[0], i);
      int checker1 = __shfl(checkers[1], i);
      if (checkers[0] == checker0 && checkers[1] == checker1) {
        valid = false;
	handled = true;
	#ifdef DEBUG
	  cpu_pointer::atomicAdd(&Critical, 1LLU);
	#endif
      }
    }

    #ifdef DEBUG
      STOP(Minor)
    #endif
  }
}

template <typename T>
__device__ inline void CPtr<T>::setInvalid()
{
  using namespace cpu_pointer;

  if (valid == true) {
    valid = false;

    #ifdef DEBUG
     START(Clear)
    #endif

    unsigned int count = 0;
    bool check = true;
    while (check) {
      int i = __ffs(__ballot(check)) - 1;
      int checker = __shfl(tag_index, i);
      bool c = tag_index == checker;
      unsigned int num  = __popc(__ballot(c));
      if (i == LANE_ID) {
        count = num;
      }
      if (c) {
        check = false;
      }
    }

    if (count) {
      cpu_pointer::atomicSub(&tag->count, count);
    }
    
    #ifdef DEBUG
      STOP(Clear)
    #endif
  }
}

template <typename T>
__device__ inline void CPtr<T>::setValues()
{
  using namespace cpu_pointer;
  page_addr = (std::uintptr_t)ptr & PAGE_MASK;
  tag_index = (page_addr >> PAGE_BIT)% d_size_cache;
  buffer = (T*)d_cache[tag_index] + (ptr - (T*)page_addr);
  tag = &d_tag[tag_index];
}

#endif // CPTR_CU_H
