
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>


#define SIZE 3
#define PAGE_ENTRIES 1024
#define GLOBAL_MEMORY 128 * 1024 // Bytes
#define SHARED_MEMORY 48 * 1024

__device__ int manage;
//__device__ __managed__ int manage;


__global__ void myKernel(int *test_d)
{
	printf("In kernel! test_d is ");

	for (int i = 0; i < SIZE; i++) {
		printf("%d ", test_d[i]);
	}
	printf("\n");

	for (int i = 0; i < SIZE; i++) {
		test_d[i] = 9;
	}

	printf("\n");
	printf("In kernel! test_d is updated as ");
	for (int i = 0; i < SIZE; i++) {
		printf("%d ", test_d[i]);
	}
	printf("\n");

	printf("In kernel! manage %d\n",manage);
	manage = 2;

}

int main()
{
	cudaError_t cudaStatus;
	int* test_h;				// host
	int* test_d;				// device
	manage = 1;
	cudaSetDevice(0);
	cudaMalloc(&test_d, sizeof(int) * SIZE);
	test_h = (int *)malloc(sizeof(int) * SIZE);

	for (int i = 0; i < SIZE; i++) {
		test_h[i] = 0;
	}

	int invert_page_table[PAGE_ENTRIES * 2];	// 0 - PageEntries-1: valid bit; pageEntries -2 * pageEntries: pageNUM


	cudaMemcpy(test_d, test_h, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
	myKernel << < 1, 1 >> > (test_d);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "my kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}

	cudaMemcpy(test_h, test_d, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);

	printf("After kernel, manage = %d\ntest_h is ",manage);
	for (int i = 0; i < SIZE; i++) {
		printf("%d ", test_h[i]);
	}
	printf("\n");

	cudaFree(test_d);
	free(test_h);

	cudaDeviceSynchronize();
	cudaDeviceReset();

	return 0;
}
