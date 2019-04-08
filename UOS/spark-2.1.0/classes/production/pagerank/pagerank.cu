#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define SHARED_SIZE 10000

/*
 * @param size          total number of links connected each page
 * @param i_src         a page referenced by other pages
 * @param i_dst         pages that reference the source page
 * @param o_pagerank    calculation result of pagerank
 * @param b_pagerank    pagerank values that is calculation result of previous iteration
 * @param b_linkcount   number of links connected each page
 * @param df            damping factor
 *
 */
extern "C"
__global__ void cuPageRank(int size, int* i_src, int* i_dst,
		float* o_pagerank, float* b_pagerank, int* b_linkcount, float df){

		const int idx = blockIdx.x*blockDim.x + threadIdx.x;

		__shared__ float s_ranks[SHARED_SIZE];


		// init shared memory
		for(int i=threadIdx.x; i<SHARED_SIZE; i+=blockDim.x) { // loop for 40?
			s_ranks[i] = 0;
		}
		__syncthreads();

		if(idx < size) {
				atomicAdd(&s_ranks[i_src[idx]], b_pagerank[i_dst[idx]] / b_linkcount[i_dst[idx]]);
				__syncthreads();

				for(int i=threadIdx.x; i<SHARED_SIZE; i+=blockDim.x) { // loop for 40?
					atomicAdd(&o_pagerank[i], df * s_ranks[i]);
				}
		}

}