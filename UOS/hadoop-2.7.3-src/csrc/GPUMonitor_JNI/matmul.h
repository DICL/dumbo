#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
//#include <cuda_runtime.h>

// For App
#define DATA_TYPE          int
#define DATA_PRINT_FORMAT  "%d "
#define DATA_RANDOM_METHOD rand()%10

#define USE_SHARED_MEMORY  true
#define TILE_WIDTH         10

// Time Measuring Function
float
getduration (struct timeval begin, struct timeval finish);

// about memory allocation and memset
void
host_malloc (DATA_TYPE** h_A, DATA_TYPE** h_B, DATA_TYPE** h_C, int M, int N);

void
host_memset (DATA_TYPE* h_A, DATA_TYPE* h_B, int M, int N);
 
void
host_free(DATA_TYPE* h_A, DATA_TYPE* h_B, DATA_TYPE* h_C);
  
void
device_malloc (DATA_TYPE** d_A, DATA_TYPE** d_B, DATA_TYPE** d_C, int M, int N);

void
device_free (DATA_TYPE* d_A, DATA_TYPE* d_B, DATA_TYPE* d_C);

void
print_matrix(char label, DATA_TYPE* matrix, int M, int N);

int
init_matmul (int m);

float
do_matmul (int m);

void
destroy_matmul ();
