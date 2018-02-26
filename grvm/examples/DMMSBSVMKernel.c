#include <cuda.h>

#include "cpuptr.h"

extern "C"
__global__ void kmain(int N, int *A,  int *B, int *C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i, row, col;
  int s, va, vb;
  cpupointer pa, pb, pc;

  if (idx >= N*N) {
    return;
  }

//  if (N <= 16) {
//    printf("kmain[%5d]:begin N=%d A=H0x%llx H0xB=%llx H0xC=%llx\n", idx, N, A, B, C);
//  }
  row = idx / N;
  col = idx % N;
  pa = cpuptr_init((char *)A + row * N * sizeof(*A));
  pb = cpuptr_init((char *)B + col * sizeof(*B));
  s = 0;
  for(i = 0; i < N;i++) {
    pa = cpuptr_read_int_fast(pa, &va);
    pb = cpuptr_read_int_fast(pb, &vb);
	  s += va * vb;
	  pa = cpuptr_inc_fast(pa, sizeof(*A));
	  pb = cpuptr_inc_fast(pb, N * sizeof(*B));
  }
  pc = cpuptr_init((char *)&C[row * N  + col]);
  pc = cpuptr_write_int(pc, s);
  cpuptr_release(pa);
  cpuptr_release(pb);
  cpuptr_release(pc);
//  if (N <= 16) {
//    printf("kmain[%5d]: written %d at H%llx\n", idx, s, &C[row * N  + col]);
//  }
}
