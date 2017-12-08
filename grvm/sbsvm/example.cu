#include "example.h"

#include <cuda.h>

#include <cstdint>
#include <cstring>
#include <cassert>

#include <iostream>

#include "helper_cuda.h"
#include "cptr.h"


__global__ void gpu_client(float (*A)[MAT_SIZE], float (*B)[MAT_SIZE], float (*C)[MAT_SIZE])
{
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;

  CPtr<float> a = A[i];
  CPtr<float> b = &B[0][j];
  CPtr<float> c = &C[i][j];

  if (i<MAT_SIZE && j<MAT_SIZE) {
    float result = 0;
    for (size_t x=0; x<MAT_SIZE; ++x) {
      float fa = a.read();
      float fb = b.read();
      result += fa * fb;
      ++a;
      B+=MAT_SIZE;
    }
    c.write(result);
  }
}

int main()
{  
  float (*A)[MAT_SIZE] = new float[MAT_SIZE][MAT_SIZE];
  float (*B)[MAT_SIZE] = new float[MAT_SIZE][MAT_SIZE];
  float (*C)[MAT_SIZE] = new float[MAT_SIZE][MAT_SIZE];
  for (size_t j=0; j<MAT_SIZE; ++j) {
    for (size_t i=0; i<MAT_SIZE; ++i) {
      C[i][j] = 0.0f;
      A[i][j]= B[i][j] = 1.0f;
    }
  }

  cpu_pointer::initialize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  size_t tries = 1;

  float sum = 0.0f;
  for (size_t i=0; i<tries; ++i) {
    dim3 block(32, 8);
    dim3 grid(MAT_SIZE/block.x+(MAT_SIZE%block.x!=0), MAT_SIZE/block.y+(MAT_SIZE%block.y!=0));

    cudaEventRecord(start);
    //cpu_pointer::prefetch(C, sizeof(**C)*MAT_SIZE*MAT_SIZE);
    //cpu_pointer::prefetch(B, sizeof(**B)*MAT_SIZE*MAT_SIZE);
    //cpu_pointer::prefetch(A, sizeof(**A)*MAT_SIZE*MAT_SIZE);
    gpu_client<<<grid, block>>>(A, B, C);
    cudaEventRecord(stop);
  
    cpu_pointer::run_handler();
     
    checkCudaErrors(cudaEventSynchronize(stop));
    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    sum += milliseconds;

    for (size_t j=0; j<MAT_SIZE; ++j) {
      for (size_t i=0; i<MAT_SIZE; ++i) {
        if (C[i][j] != MAT_SIZE) {
          fprintf(stderr, "(%p)C[%zu][%zu]=%f\n",&C[i][j],i,j,C[i][j]);
	  // assert(!"wrong result");
	}
      }
    }

    cpu_pointer::clear_cache();
  }
  std::cout << MAT_SIZE << "k x " << MAT_SIZE << "k : " <<  sum/tries << " ms" << std::endl;

  cpu_pointer::finalize();

  delete[] C;
  delete[] B;
  delete[] A;
  
  return 0;
}
