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
		cudaEventDestroy(kernelEvent);\
		free(streams);\
		free(deviceProp);\
		return -1; \
	} \
}


	__global__
void kernelAdd()
{
#if 0
	clock_t start_clock = clock();
	clock_t clock_offset = 0;
	while (clock_offset < 40000000)
	{
		clock_offset = clock() - start_clock;
	}
#endif
	//*num += inc;
}

	__global__
void kernelMult()
{
	//printf("*num: %d, *mult: %d\n", *num, *mult);
	//*num *= *mult;
	//printf("*num: %d, *mult: %d\n", *num, *mult);
}

int main(int argc, char *argv[])
{
	if(argc != 2) {
		fprintf(stderr, "usage: ./bench <thread-count>\n");
		return -1;
	}
	int ngpus = atoi(argv[1]);
	int nstreams = ngpus + 1;
	if(ngpus <= 0) {
		fprintf(stderr, "thread count is supposed to be a nonzero integer\n");
		return -1;
	}
	uint64_t diff;
	struct timespec start, end;

	int num_gpus = 0;

	GUARD_CUDACALL(cudaGetDeviceCount(&num_gpus), "cudaGetDeviceCount", __LINE__);

	if(ngpus > num_gpus) {
		fprintf(stderr, "Thread count cannot be higher than the number of available GPUs.\n");
		return -1;
	}
	//#if 0
	cudaStream_t *streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));

	cudaEvent_t kernelEvent;
	cudaEvent_t * eventStart = (cudaEvent_t *) malloc(ngpus * sizeof(cudaDeviceProp)); 
	cudaEvent_t * eventStop = (cudaEvent_t *) malloc(ngpus * sizeof(cudaDeviceProp));

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
		//#if 0
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		for(int j = 0; j < ngpus; j++) {
			if(i != j) {
				GUARD_CUDACALL2(cudaDeviceEnablePeerAccess(j, 0), "cudaDeviceEnablePeerAccess", __LINE__);
			}
		}
		//#endif
	}
	for(int i = 0; i < nstreams; i++) {
		if(i == 0 || i == 1)
                        cudaSetDevice(0);
                else
                        cudaSetDevice(i - 1);
		GUARD_CUDACALL2(cudaStreamCreate(&(streams[i])), "cudaStreamCreate", __LINE__);
	}

	GUARD_CUDACALL2(cudaSetDevice(0), "cudaSetDevice", __LINE__);
	cudaEventCreateWithFlags(&kernelEvent, cudaEventDisableTiming);
	for(int i = 0; i < ngpus; i++) {
		cudaSetDevice(i);
		cudaEventCreate(&eventStart[i]);
        	cudaEventCreate(&eventStop[i]);
        }		

	cudaSetDevice(0);
	clock_gettime(CLOCK_MONOTONIC, &start);       /* mark start time */
	for(int i = 0; i < ngpus; i++) {
		cudaSetDevice(i);
		cudaEventRecord(eventStart[i], streams[i+1]);
	}
	cudaSetDevice(0);
	kernelAdd<<<1, 1, 0, streams[0]>>>();
	cudaEventRecord(kernelEvent, streams[0]);
	//kernelMult<<<1, 1, 0, streams[0]>>>(num_d, num_1_d);
	//cudaMemcpyAsync(num_1, num_1_d, sizeof(int), cudaMemcpyDeviceToHost, streams[0]);
	for(int i = 0; i < ngpus; i++) {	
		cudaSetDevice(i);
		cudaStreamWaitEvent(streams[i+1], kernelEvent,0);
		//cudaMemcpyAsync(nums_d[i], nums[i], sizeof(int), cudaMemcpyHostToDevice, streams[i+1]);
		//kernelMult<<<1, 1, 0, streams[i+1]>>>(num_d, nums_d[i]);
		//cudaMemcpyAsync(nums[i], nums_d[i], sizeof(int), cudaMemcpyDeviceToHost, streams[i+1]);
	}
	for(int i = 0; i < ngpus; i++) {
		cudaSetDevice(i);
                cudaEventRecord(eventStop[i], streams[i+1]);
        }
	for(int i = 0; i < ngpus; i++) {
                cudaSetDevice(i);
                cudaEventSynchronize(eventStop[i]);
        }
	float * milliseconds = (float *) malloc(ngpus * sizeof(float));

	for(int i = 0; i < ngpus; i++) {
		cudaEventElapsedTime(&milliseconds[i], eventStart[i], eventStop[i]);
		printf("elapsed time in gpu %d = %0.2f nanoseconds\n", i, milliseconds[i] * 1000000);
	}
	for(int i = 0; i < nstreams; i++) {
		if(i == 0 || i == 1)
			cudaSetDevice(0);
		else
			cudaSetDevice(i - 1);
		cudaStreamSynchronize(streams[i]);
	}
	//cudaSetDevice(0);
	clock_gettime(CLOCK_MONOTONIC, &end); /* mark the end time */
	diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
#if 0
	printf("num_1: %d\n", *num_1);
	for(int i = 0; i < ngpus - 1; i++)
		printf("num_%d: %d\n", i+2, *nums[i]);
#endif
	for(int i = 0; i < nstreams; i++) {
		if(i == 0 || i == 1)
                        cudaSetDevice(0);
                else
                        cudaSetDevice(i - 1);		
		cudaStreamDestroy(streams[i]);
		if(i == 0) {
			cudaEventDestroy(kernelEvent);	
		}
	}		

	cudaSetDevice(0);

	free(streams);
	//free(kernelEvent);
	return 0;
}
