#include <stdio.h>
#include <time.h>
#include <cuda.h>

#define CHKCUDA(val)  ({                                         \
            cudaError_t v = (val);                               \
  if (v != cudaSuccess) {                                        \
      const char *ename;                                         \
      ename = cudaGetErrorName(v);                               \
      fprintf(stderr, "CUDA error %s(%d) at %s:%d\n",            \
              ename, v, __FILE__, __LINE__);                     \
      exit(EXIT_FAILURE);                                        \
  }                                                              \
  v;})

__global__ void kmain(int N, int *A, int *B, int *C) {
    int idx;
    int row, col;
    int s, i;

    idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) {
        return;
    }
    row = idx / N;
    col = idx % N;

    for( s = 0, i = 0; i < N;i++) {
        s += A[row * N + i] * B[ i * N + col];
    }
    C[row * N + col] = s;
}


int main(int argc, char **argv) {
  int N;
  int *A, *B, *C;
  int blocks;
  int *dA, *dB, *dC;
  int asize;
  clock_t ts;
  
  N = (argc > 1) ? atoi(argv[1]) : 512;
  asize = N * N * sizeof (int);
  A = (int *)malloc(asize);
  B = (int *)malloc(asize);
  C = (int *)malloc(asize);

  printf("=== begin %s where N  is %d === \n", argv[0], N);
  ts = clock();
  CHKCUDA(cudaMalloc(&dA, asize));
  CHKCUDA(cudaMalloc(&dB, asize));
  CHKCUDA(cudaMalloc(&dC, asize));
  CHKCUDA(cudaMemcpy(dA, A, asize, cudaMemcpyHostToDevice));
  CHKCUDA(cudaMemcpy(dB, B, asize, cudaMemcpyHostToDevice));
  CHKCUDA(cudaMemcpy(dC, C, asize, cudaMemcpyHostToDevice));
  blocks = N * N / 512;
  kmain<<<blocks, 512>>>(N, dA, dB, dC);
  CHKCUDA(cudaDeviceSynchronize());
  CHKCUDA(cudaMemcpy(A, dA, asize, cudaMemcpyDeviceToHost));
  CHKCUDA(cudaMemcpy(B, dB, asize, cudaMemcpyDeviceToHost));
  CHKCUDA(cudaMemcpy(C, dC, asize, cudaMemcpyDeviceToHost));  

  CHKCUDA(cudaFree(dA));
  CHKCUDA(cudaFree(dB));
  CHKCUDA(cudaFree(dC));
  printf("=== finished %s in %d ms ===\n",
         argv[0],
         (int)((clock() - ts) / (CLOCKS_PER_SEC/1000)));
  free(A);
  free(B);
  free(C);

  return 0;
}
