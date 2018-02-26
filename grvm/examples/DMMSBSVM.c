#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

#include "sbsvm.h"
#include "DMMSBSVMKernel.ptx.h"

int main(int argc, char *argv[]) {
  int N, *A, *B, *C;
  clock_t ts;
  sbsvmcontext cxt;
  void * kparams[4];

  if (argc >= 2) {
    N = atoi(argv[1]);
  } else {
    N = 512;
  }

  A = (int *)malloc(N * N * sizeof(int));
  B = (int *)malloc(N * N * sizeof(int));
  C = (int *)malloc(N * N * sizeof(int));

  printf("=== begin %s where N  is %d === \n", argv[0], N);
  printf("A = H0x%lx-H0x%lx\n", (uintptr_t)A, (uintptr_t)A + N * N * sizeof(int));
  printf("B = H0x%lx-H0x%lx\n", (uintptr_t)B, (uintptr_t)B + N * N * sizeof(int));
  printf("C = H0x%lx-H0x%lx\n", (uintptr_t)C, (uintptr_t)C + N * N * sizeof(int));
  ts = clock();
  cxt = sbsvm_open(1024);
  kparams[0] = &N;
  kparams[1] = &A;
  kparams[2] = &B;
  kparams[3] = &C;
//  printf("A=0x%lx B = 0x%lx C = 0x%lx\n", (uintptr_t)A, (uintptr_t)B, (uintptr_t)C);
  sbsvm_execute(cxt,
      dmm_kernel_ptx, sizeof(dmm_kernel_ptx),
      0, N * N,  kparams);
  sbsvm_close(cxt);
  printf("=== finished %s in %d ms ===\n",
      argv[0],
		  (int)((clock() - ts) / (CLOCKS_PER_SEC/1000)));
}
