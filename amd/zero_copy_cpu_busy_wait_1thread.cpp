#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdint.h>     /* for uint64 definition */
#include <stdlib.h>     /* for exit() definition */
#include <time.h>       /* for clock_gettime */
#include <omp.h>
#include <iostream>
#include <unistd.h>

#define BILLION 1000000000L

using namespace std;

hipError_t errorHandler(hipError_t error, const char * func_name, int line) {
	if(error != hipSuccess)
		fprintf(stderr, "%s returned error %s, line(%d)\n", func_name, hipGetErrorString(error),line);
	return error;
}

hipError_t err;

#define GUARD_CUDACALL(cuda_call, message, line)\
{       \
	err = errorHandler(cuda_call, message, line); \
	if (err != hipSuccess) { \
		return -1; \
	} \
}

#define GUARD_CUDACALL2(cuda_call, message, line)\
{       \
	err = errorHandler(cuda_call, message, line); \
	if (err != hipSuccess) { \
		free(streams);\
		free(deviceProp);\
		return -1; \
	} \
}

	__global__
void kernel1(clock_t clock_count, int stream)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	clock_t start_clock = clock();
	clock_t clock_offset = 0;
	printf("thread with id %d is in stream %d starts\n", i, stream);
	while (clock_offset < clock_count)
	{
		clock_offset = clock() - start_clock;
	} 
	printf("thread with id %d is in stream %d ends\n", i, stream);
}

	__global__
void kernelAdd(int inc, int * num, int * mult, int ngpus, volatile int * cpu_flag_pointer)
{
#if 0
	clock_t start_clock = clock();
	clock_t clock_offset = 0;
	while (clock_offset < 40000)
	{
		clock_offset = clock() - start_clock;
	}
#endif
	*num += inc;
	*cpu_flag_pointer = 1;
	*mult *= *num; 
}

	__global__
void kernelMult(int * mult, int * num)
{
	*num *= *mult;
}

int main(int argc, char *argv[])
{
	if(argc != 2) {
		fprintf(stderr, "usage: ./bench <thread-count>\n");
		return -1;
	}
	int ngpus = atoi(argv[1]);
	if(ngpus <= 0) {
		fprintf(stderr, "GPU count is supposed to be a nonzero integer\n");
		return -1;
	}
	uint64_t diff;
	struct timespec start, end;

	int num_gpus = 0;

	cout << "chk 1\n";
	GUARD_CUDACALL(hipGetDeviceCount(&num_gpus), "hipGetDeviceCount", __LINE__);

	if(ngpus > num_gpus) {
		fprintf(stderr, "GPU count cannot be higher than the number of available GPUs.\n");
		return -1;
	}
	//#if 0
	hipStream_t *streams = (hipStream_t *) malloc(ngpus * sizeof(hipStream_t));

	hipDeviceProp_t *deviceProp = (hipDeviceProp_t *) malloc(ngpus * sizeof(hipDeviceProp_t));
 
	cout << "chk 2\n";

	for(int i = 0; i < ngpus; i++) {
		GUARD_CUDACALL2(hipSetDevice(i), "hipSetDevice", __LINE__);
		GUARD_CUDACALL2(hipGetDeviceProperties(&deviceProp[i], i), "hipGetDeviceProperties", __LINE__);
		if (!deviceProp[i].canMapHostMemory) {
			fprintf(stderr, "GPU %d does not support mapping CPU host memory!\n", i);
			free(streams);
			free(deviceProp);
			return -1;
		}

		for(int j = 0; j < ngpus; j++) {
			if(i != j) {
				GUARD_CUDACALL2(hipDeviceEnablePeerAccess(j, 0), "hipDeviceEnablePeerAccess", __LINE__);
			}
		}

		GUARD_CUDACALL2(hipStreamCreate(&(streams[i])), "hipStreamCreate", __LINE__);
	}

	cout << "chk 3\n";

	GUARD_CUDACALL2(hipSetDevice(0), "hipSetDevice", __LINE__);
	int * cpu_flag;
	//hipHostAlloc((void **)&cpu_flag, sizeof(int), hipHostMallocMapped);	
	hipHostMalloc((void**)&cpu_flag, sizeof(uint32_t), hipHostMallocDefault );

	int * num_h;
	hipHostMalloc(&num_h, sizeof(int));
	*num_h = 1;
	int * num_d;
	int * num_1;
	hipHostMalloc(&num_1, sizeof(int));
	*num_1 = 2;
	int * num_1_d;
	hipMalloc(&num_1_d, sizeof(int));

	int ** nums = (int **) malloc ((ngpus - 1) * sizeof(int*));
	int ** nums_d = (int **) malloc ((ngpus - 1) * sizeof(int*));

	int *cpu_flag_pointer; //= (int **) malloc (sizeof(int *));
	*cpu_flag = 0;	

	cout << "chk 4\n";
	//hipSetDevice(0);
	hipMalloc(&num_d, sizeof(int));
	hipMemcpyAsync(num_d, num_h, sizeof(int), hipMemcpyHostToDevice, streams[0]);
	hipMemcpyAsync(num_1_d, num_1, sizeof(int), hipMemcpyHostToDevice, streams[0]);

	hipHostGetDevicePointer((void **)&cpu_flag_pointer, (void *)cpu_flag, 0);
	for(int i = 0; i < ngpus - 1; i++) {
		hipSetDevice(i+1);
		hipHostMalloc(&nums[i], sizeof(int));
		*nums[i] = 3;
		hipMalloc(&nums_d[i], sizeof(int));
	}

	cout << "chk 5\n";
	hipSetDevice(0);
	clock_gettime(CLOCK_MONOTONIC, &start); 
	hipLaunchKernelGGL(kernelAdd, 1, 1, 0, streams[0], 2, num_d, num_1_d, ngpus, cpu_flag_pointer);
	//hipMemcpyAsync(num_1, num_1_d, sizeof(int), hipMemcpyDeviceToHost, streams[0]);
	//hipEventRecord(kernelEvent[0], streams[0]);
	//sleep(0.0001);
	while(*cpu_flag == 0);
	if(ngpus > 1) {
		for(int i = 1; i < ngpus; i++) {
			hipSetDevice(i);
			//hipMemcpyAsync(nums_d[i-1], nums[i-1], sizeof(int), hipMemcpyHostToDevice, streams[i]);
			hipLaunchKernelGGL(kernelMult, 1, 1, 0, streams[i], num_d, nums_d[i-1]);
			//hipMemcpyAsync(nums[i-1], nums_d[i-1], sizeof(int), hipMemcpyDeviceToHost, streams[i]);
		}
	}
	cout << "chk 6\n";
	for(int i = 0; i < ngpus; i++) {
		hipStreamSynchronize(streams[i]);
	}
	clock_gettime(CLOCK_MONOTONIC, &end); /* mark the end time */
	diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
	printf("num_1: %d\n", *num_1);
	for(int i = 0; i < ngpus - 1; i++)
		printf("num_%d: %d\n", i+2, *nums[i]);

	for(int i = 0; i < ngpus; i++) {
		hipSetDevice(i);
		hipStreamDestroy(streams[i]);
	}

	hipSetDevice(0);

	for(int i = 0; i < ngpus - 1; i++) {
		hipSetDevice(i+1);
		hipHostFree(nums[i]);
		hipFree(nums_d[i]);
	}

	free(nums);
	free(nums_d);

	hipHostFree(num_h);
	//hipHostFree(cpu_flag);
	free(cpu_flag_pointer);
	hipFree(num_d);

	hipHostFree(num_1);
	hipFree(num_1_d);

	free(streams);
	//free(kernelEvent);
	return 0;
}
