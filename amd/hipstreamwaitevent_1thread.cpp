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
		hipEventDestroy(kernelEvent);\
		free(streams);\
		free(deviceProp);\
		return -1; \
	} \
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
	int nstreams = ngpus + 1;
	if(ngpus <= 0) {
		fprintf(stderr, "thread count is supposed to be a nonzero integer\n");
		return -1;
	}
	uint64_t diff;
	struct timespec start, end;

	int num_gpus = 0;

	GUARD_CUDACALL(hipGetDeviceCount(&num_gpus), "hipGetDeviceCount", __LINE__);

	if(ngpus > num_gpus) {
		fprintf(stderr, "Thread count cannot be higher than the number of available GPUs.\n");
		return -1;
	}
	//#if 0
	hipStream_t *streams = (hipStream_t *) malloc(ngpus * sizeof(hipStream_t));

	hipEvent_t kernelEvent;
	hipEvent_t * eventStart = (hipEvent_t *) malloc(ngpus * sizeof(hipEvent_t));
        hipEvent_t * eventStop = (hipEvent_t *) malloc(ngpus * sizeof(hipEvent_t));	

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
		//#if 0
		hipSetDeviceFlags(hipDeviceScheduleBlockingSync);
		for(int j = 0; j < ngpus; j++) {
			if(i != j) {
				GUARD_CUDACALL2(hipDeviceEnablePeerAccess(j, 0), "hipDeviceEnablePeerAccess", __LINE__);
			}
		}
		//#endif
	}

	for(int i = 0; i < nstreams; i++) {
                if(i == 0 || i == 1)
                        hipSetDevice(0);
                else
                        hipSetDevice(i - 1);
                GUARD_CUDACALL2(hipStreamCreate(&(streams[i])), "hipStreamCreate", __LINE__);
        }	

	GUARD_CUDACALL2(hipSetDevice(0), "hipSetDevice", __LINE__);
	hipEventCreateWithFlags(&kernelEvent, hipEventDisableTiming);
	for(int i = 0; i < ngpus; i++) {
                hipSetDevice(i);
                hipEventCreate(&eventStart[i]);
                hipEventCreate(&eventStop[i]);
        }	


	clock_gettime(CLOCK_MONOTONIC, &start);       /* mark start time */
	for(int i = 0; i < ngpus; i++) {
                hipSetDevice(i);
                hipEventRecord(eventStart[i], streams[i+1]);
        }
	hipSetDevice(0);
	hipLaunchKernelGGL(kernelAdd, dim3(1), dim3(1), 0, streams[0]);
	hipEventRecord(kernelEvent, streams[0]);
	//hipLaunchKernelGGL(kernelMult, 1, 1, 0, streams[0], num_d, num_1_d);
	//hipMemcpyAsync(num_1, num_1_d, sizeof(int), hipMemcpyDeviceToHost, streams[0]);	
	for(int i = 0; i < ngpus; i++) {
                hipSetDevice(i);
                hipStreamWaitEvent(streams[i+1], kernelEvent,0);
                //cudaMemcpyAsync(nums_d[i], nums[i], sizeof(int), cudaMemcpyHostToDevice, streams[i+1]);
                //kernelMult<<<1, 1, 0, streams[i+1]>>>(num_d, nums_d[i]);
                //cudaMemcpyAsync(nums[i], nums_d[i], sizeof(int), cudaMemcpyDeviceToHost, streams[i+1]);
        }	
//#endif
	for(int i = 0; i < ngpus; i++) {
                hipSetDevice(i);
                hipEventRecord(eventStop[i], streams[i+1]);
        }
	for(int i = 0; i < ngpus; i++) {
                hipSetDevice(i);
                hipEventSynchronize(eventStop[i]);
        }	
	float * milliseconds = (float *) malloc(ngpus * sizeof(float));

        for(int i = 0; i < ngpus; i++) {
                hipEventElapsedTime(&milliseconds[i], eventStart[i], eventStop[i]);
                printf("elapsed time in gpu %d = %0.2f nanoseconds\n", i, milliseconds[i] * 1000000);
        }		
	for(int i = 0; i < nstreams; i++) {
                if(i == 0 || i == 1)
                        hipSetDevice(0);
                else
                        hipSetDevice(i - 1);
                hipStreamSynchronize(streams[i]);
        }

	//hipSetDevice(0);
	clock_gettime(CLOCK_MONOTONIC, &end); /* mark the end time */
	diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);

	for(int i = 0; i < nstreams; i++) {
                if(i == 0 || i == 1)
                        hipSetDevice(0);
                else
                        hipSetDevice(i - 1);
                hipStreamDestroy(streams[i]);
                if(i == 0) {
                        hipEventDestroy(kernelEvent);
                }
        }	

	for(int i = 0; i < ngpus; i++) {
                hipSetDevice(i);
                hipEventDestroy(eventStart[i]);
                hipEventDestroy(eventStop[i]);
        }	

	free(streams);
	//free(kernelEvent);
	return 0;
}
