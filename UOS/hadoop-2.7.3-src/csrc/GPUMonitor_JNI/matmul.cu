#include "matmul.h"

//
#if ( DEBUG == true )
  #define dprintf(...) printf(__VA_ARGS__)
#else
  #define dprintf(...) 
#endif

void
print_usage (const char* exe) {
    printf("usage: %s <M> <N> \t(M,N must be multiple of %d)\n", exe, TILE_WIDTH);
}

/**
 * CUDA Kernel Device code
 */

// with NO shared memory
__global__ void
matrixMultiplication
(const DATA_TYPE *A, const DATA_TYPE *B, DATA_TYPE *C, const int M, const int N)
{
    // get id
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    DATA_TYPE sum = 0;

    for (int i=0; i<N; ++i)
	sum += A[row*N + i] * B[col + i*M];

    C[row*M + col] = sum;
}


// with shared memory
__global__ void
matrixMultiplicationSharedMem
(const DATA_TYPE *A, const DATA_TYPE *B, DATA_TYPE *C, const int M, const int N)
{
    __shared__ DATA_TYPE As[TILE_WIDTH][TILE_WIDTH];
    __shared__ DATA_TYPE Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by*TILE_WIDTH + ty;
    int col = bx*TILE_WIDTH + tx;

    DATA_TYPE value = 0;

    for (int m=0; m<N/TILE_WIDTH; ++m) {
	As [ty][tx] = A [ row*N + (m*TILE_WIDTH + tx) ];
	Bs [ty][tx] = B [ (m*TILE_WIDTH + ty)*M + col ];

	__syncthreads();

	for (int k=0; k<TILE_WIDTH; ++k) {
	    value += As[ty][k]*Bs[k][tx];
	}

	__syncthreads();
    }

    C[row*M+col] = value;
}


// Time Measuring Function
float
getduration (struct timeval begin, struct timeval finish)
{
    return ((finish.tv_sec-begin.tv_sec)*1000
	    +(float)(finish.tv_usec-begin.tv_usec)/1000); // ms
}

// about memory allocation and memset
void
host_malloc (DATA_TYPE** h_A, DATA_TYPE** h_B, DATA_TYPE** h_C, int M, int N) {
    *h_A = (DATA_TYPE*) malloc( sizeof(DATA_TYPE) * M * N);
    *h_B = (DATA_TYPE*) malloc( sizeof(DATA_TYPE) * N * M);
    *h_C = (DATA_TYPE*) malloc( sizeof(DATA_TYPE) * M * M);
}

void
host_memset (DATA_TYPE* h_A, DATA_TYPE* h_B, int M, int N) {
//    srand(time(NULL));
//    for (int i=0; i<(M*N); i++) {
//	h_A[i] = (DATA_TYPE)(DATA_RANDOM_METHOD);
//	h_B[i] = (DATA_TYPE)(DATA_RANDOM_METHOD);
//    }
    memset (h_A, 1, M*N);
    memset (h_B, 1, M*N);
}
 
void
host_free(DATA_TYPE* h_A, DATA_TYPE* h_B, DATA_TYPE* h_C) {
    free(h_A);
    free(h_B);
    free(h_C);
}
  
void
device_malloc (DATA_TYPE** d_A, DATA_TYPE** d_B, DATA_TYPE** d_C, int M, int N) {
    cudaMalloc((void **)d_A, sizeof(DATA_TYPE) * M * N);
    cudaMalloc((void **)d_B, sizeof(DATA_TYPE) * N * M);
    cudaMalloc((void **)d_C, sizeof(DATA_TYPE) * M * M);
}

void
device_free (DATA_TYPE* d_A, DATA_TYPE* d_B, DATA_TYPE* d_C) {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/**
 * Host main routine
 */    

struct timeval begin, finish;
dim3           blocks, threads;
DATA_TYPE      *h_A, *h_B, *h_C;
DATA_TYPE      *d_A, *d_B, *d_C;

int
init_matmul (int m)
{
    if (m%TILE_WIDTH) {
	return -1;
    }
    int n = m;

    blocks  = dim3(m/TILE_WIDTH, n/TILE_WIDTH);
    threads = dim3(  TILE_WIDTH,   TILE_WIDTH);
    
    // Host Matrix
    host_malloc (&h_A, &h_B, &h_C, m, n);
    host_memset (h_A, h_B, m, n);
    
    // Device Matrix
    device_malloc (&d_A, &d_B, &d_C, m, n);
    
    // Copy the Host Matrices to Device
    cudaMemcpy (d_A, h_A, sizeof(DATA_TYPE) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy (d_B, h_B, sizeof(DATA_TYPE) * n * m, cudaMemcpyHostToDevice);

    return 0;
}
    
float
do_matmul (int m)
{

    if (m%TILE_WIDTH) {
	return -1;
    }
    
    int n = m;
    gettimeofday(&begin, NULL);

    // kernel launch
#if ( USE_SHARED_MEMORY == true )
    matrixMultiplicationSharedMem <<<blocks, threads>>>(d_A, d_B, d_C, m, n);
#else
    matrixMultiplication	  <<<blocks, threads>>>(d_A, d_B, d_C, m, n);
#endif
    cudaDeviceSynchronize();

    gettimeofday(&finish, NULL);

    return getduration(begin, finish);
}

void
destroy_matmul ()
{
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
