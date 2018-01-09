#ifndef EXAMPLE_H
#define EXAMPLE_H

#define MAT_SIZE (1*1024ul)

__global__ void gpu_client(float (*A)[MAT_SIZE], float (*B)[MAT_SIZE], float (*C)[MAT_SIZE]);

#endif // EXAMPLE_H
