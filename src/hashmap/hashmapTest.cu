#include <cuda.h>
#include <stdio.h>
#include "hash_map_template.h"
using namespace CUDASTL;

__global__ void foo(HashMap<int, int> * h) {
	int i = get_thread_id();
	//	int i=threadIdx.x;
	(*h)[i] = i * 10;
}

__global__ void bar(HashMap<int, int> * h, int * arr) {
	//	int i=get_thread_id();
	//	arr[i]=(*h)[i];
	for (HashMap<int, int>::Iterator it = h->begin(); it != h->end(); ++it) {
		arr[it->key] = it->value;
	}

}

const int numBlocks = 10;
const int numThreads = 10;
const int N = numThreads*numBlocks;

int main() {

	int * d_arr;
	cudaMalloc((void **)&d_arr, sizeof(int)*N);
	HashMap<int, int> * h = CreateHashMap<int, int, HashFunc<int> >(79, 2000);

	foo << <numBlocks, numThreads >> > (h);

	cudaThreadSynchronize();
	//	bar<<<numBlocks,numThreads>>>(h,d_arr);
	bar << <1, 1 >> >(h, d_arr);
	int * h_arr = (int *)malloc(sizeof(int)*N);
	cudaMemcpy(h_arr, d_arr, sizeof(int)*N, cudaMemcpyDeviceToHost);

	int i;
	for (i = 90; i<100; i++) {
		printf("%d\t", h_arr[i]);
		if (i % 8 == 7)
			printf("\n");
	}
	printf("\n");

	return 0;
}