#include <stdio.h>
#include <stdint.h>     /* for uint64 definition */
#include <stdlib.h>     /* for exit() definition */
#include <time.h>       /* for clock_gettime */
#include <omp.h>

#define BILLION 1000000000L

cudaError_t errorHandler(cudaError_t error, const char * func_name, int line) {
	if(error != cudaSuccess)
		fprintf(stderr, "%s returned error %s, line(%d)\n", func_name, cudaGetErrorString(error),line);
	return error;
}

cudaError_t err;

#define GUARD_CUDACALL(cuda_call, message, line)\
{       \
	err = errorHandler(cuda_call, message, line); \
	if (err != cudaSuccess) { \
		return -1; \
	} \
}

#define GUARD_CUDACALL2(cuda_call, message, line)\
{       \
	err = errorHandler(cuda_call, message, line); \
	if (err != cudaSuccess) { \
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

	GUARD_CUDACALL(cudaGetDeviceCount(&num_gpus), "cudaGetDeviceCount", __LINE__);

	if(ngpus > num_gpus) {
		fprintf(stderr, "GPU count cannot be higher than the number of available GPUs.\n");
		return -1;
	}
	//#if 0
	cudaStream_t *streams = (cudaStream_t *) malloc(ngpus * sizeof(cudaStream_t));

	cudaDeviceProp *deviceProp = (cudaDeviceProp *) malloc(ngpus * sizeof(cudaDeviceProp));


	for(int i = 0; i < ngpus; i++) {
		GUARD_CUDACALL2(cudaSetDevice(i), "cudaSetDevice", __LINE__);
		GUARD_CUDACALL2(cudaGetDeviceProperties(&deviceProp[i], i), "cudaGetDeviceProperties", __LINE__);
		if (!deviceProp[i].canMapHostMemory) {
			fprintf(stderr, "GPU %d does not support mapping CPU host memory!\n", i);
			free(streams);
			free(deviceProp);
			return -1;
		}
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		for(int j = 0; j < ngpus; j++) {
			if(i != j) {
				GUARD_CUDACALL2(cudaDeviceEnablePeerAccess(j, 0), "cudaDeviceEnablePeerAccess", __LINE__);
			}
		}

		GUARD_CUDACALL2(cudaStreamCreate(&(streams[i])), "cudaStreamCreate", __LINE__);
	}


	GUARD_CUDACALL2(cudaSetDevice(0), "cudaSetDevice", __LINE__);

	clock_gettime(CLOCK_MONOTONIC, &start); 
	kernelAdd<<<1, 1, 0, streams[0]>>>();
	//cudaMemcpyAsync(num_1, num_1_d, sizeof(int), cudaMemcpyDeviceToHost, streams[0]);
	//cudaEventRecord(kernelEvent[0], streams[0]);

	//while(*cpu_flag == 0);
	for(int i = 0; i < ngpus; i++) {
                cudaSetDevice(i);
                cudaStreamSynchronize(streams[i]);
        }
	//while(*cpu_flag == 0);	
	//if(ngpus > 1) {
#if 0
	for(int i = 0; i < ngpus; i++) {
		cudaSetDevice(i);
		//cudaMemcpyAsync(nums_d[i-1], nums[i-1], sizeof(int), cudaMemcpyHostToDevice, streams[i]);
		kernelMult<<<1, 1, 0, streams[i]>>>(num_d, nums_d[i-1]);
		//cudaMemcpyAsync(nums[i-1], nums_d[i-1], sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
	}
#endif
#if 0
	for(int i = 0; i < ngpus; i++) {
		cudaSetDevice(i);
		cudaStreamSynchronize(streams[i]);
	}
#endif
	clock_gettime(CLOCK_MONOTONIC, &end); /* mark the end time */
	diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
	for(int i = 0; i < ngpus; i++) {
		cudaSetDevice(i);
		cudaStreamDestroy(streams[i]);
	}

	free(streams);
	//free(kernelEvent);
	return 0;
}
