#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdint.h>     /* for uint64 definition */
#include <stdlib.h>     /* for exit() definition */
#include <time.h>       /* for clock_gettime */
#include <omp.h>

#define BILLION 1000000000L

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
void kernelAdd()
{
}

	__global__
void kernelMult()
{
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

	GUARD_CUDACALL(hipGetDeviceCount(&num_gpus), "hipGetDeviceCount", __LINE__);

	if(ngpus > num_gpus) {
		fprintf(stderr, "GPU count cannot be higher than the number of available GPUs.\n");
		return -1;
	}
	//#if 0
	hipStream_t *streams = (hipStream_t *) malloc(ngpus * sizeof(hipStream_t));

	hipDeviceProp_t *deviceProp = (hipDeviceProp_t *) malloc(ngpus * sizeof(hipDeviceProp_t));


	for(int i = 0; i < ngpus; i++) {
		GUARD_CUDACALL2(hipSetDevice(i), "hipSetDevice", __LINE__);
		GUARD_CUDACALL2(hipGetDeviceProperties(&deviceProp[i], i), "hipGetDeviceProperties", __LINE__);
		if (!deviceProp[i].canMapHostMemory) {
			fprintf(stderr, "GPU %d does not support mapping CPU host memory!\n", i);
			free(streams);
			free(deviceProp);
			return -1;
		}
		hipSetDeviceFlags(hipDeviceScheduleBlockingSync);
		for(int j = 0; j < ngpus; j++) {
			if(i != j) {
				GUARD_CUDACALL2(hipDeviceEnablePeerAccess(j, 0), "hipDeviceEnablePeerAccess", __LINE__);
			}
		}

		GUARD_CUDACALL2(hipStreamCreate(&(streams[i])), "hipStreamCreate", __LINE__);
	}


	GUARD_CUDACALL2(hipSetDevice(0), "hipSetDevice", __LINE__);

	hipSetDevice(0);
	clock_gettime(CLOCK_MONOTONIC, &start); 
	hipLaunchKernelGGL(kernelAdd, dim3(1), dim3(1), 0, streams[0]);

	for(int i = 0; i < ngpus; i++) {
                hipSetDevice(i);
                hipStreamSynchronize(streams[i]);
        }
	clock_gettime(CLOCK_MONOTONIC, &end); /* mark the end time */
	diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
	for(int i = 0; i < ngpus; i++) {
		hipSetDevice(i);
		hipStreamDestroy(streams[i]);
	}
	free(streams);
	//free(kernelEvent);
	return 0;
}
