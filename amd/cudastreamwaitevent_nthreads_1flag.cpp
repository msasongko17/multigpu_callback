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
void kernelAdd(int inc, volatile int * num)
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
}

__global__
void kernelMult(volatile int * mult, volatile int * num)
{
	//printf("*num: %d, *mult: %d\n", *num, *mult);
        *num *= *mult;
	//printf("*num: %d, *mult: %d\n", *num, *mult);
}

int main(int argc, char *argv[])
{
        if(argc != 2) {
                fprintf(stderr, "usage: ./bench <thread-count>\n");
                return -1;
        }
        int nthreads = atoi(argv[1]);
        if(nthreads <= 0) {
                fprintf(stderr, "thread count is supposed to be a nonzero integer\n");
                return -1;
        }
        uint64_t diff;
        struct timespec start, end;

        int num_gpus = 0;

        GUARD_CUDACALL(hipGetDeviceCount(&num_gpus), "hipGetDeviceCount", __LINE__);

        if(nthreads > num_gpus) {
                fprintf(stderr, "Thread count cannot be higher than the number of available GPUs.\n");
                return -1;
        }
        //#if 0
        hipStream_t *streams = (hipStream_t *) malloc(nthreads * sizeof(hipStream_t));

	hipEvent_t kernelEvent;

        hipDeviceProp_t *deviceProp = (hipDeviceProp_t *) malloc(nthreads * sizeof(hipDeviceProp_t));


        for(int i = 0; i < nthreads; i++) {
                GUARD_CUDACALL2(hipSetDevice(i), "hipSetDevice", __LINE__);
                GUARD_CUDACALL2(hipGetDeviceProperties(&deviceProp[i], i), "hipGetDeviceProperties", __LINE__);
                if (!deviceProp[i].canMapHostMemory) {
                        fprintf(stderr, "GPU %d does not support mapping CPU host memory!\n", i);
                        free(streams);
                        free(deviceProp);
                        return -1;
                }
//#if 0
                for(int j = 0; j < nthreads; j++) {
                        if(i != j) {
                                GUARD_CUDACALL2(hipDeviceEnablePeerAccess(j, 0), "hipDeviceEnablePeerAccess", __LINE__);
                        }
                }
		GUARD_CUDACALL2(hipStreamCreate(&(streams[i])), "hipStreamCreate", __LINE__);
//#endif
        }

	GUARD_CUDACALL2(hipSetDevice(0), "hipSetDevice", __LINE__);
	hipEventCreateWithFlags(&kernelEvent, hipEventDisableTiming);

        int * num_h;
        hipHostMalloc(&num_h, sizeof(int));
        *num_h = 1;
        int * num_d;
        int * num_1;
        hipHostMalloc(&num_1, sizeof(int));
        *num_1 = 2;
        int * num_1_d;
        hipMalloc(&num_1_d, sizeof(int));

        int ** nums = (int **) malloc ((nthreads - 1) * sizeof(int*));
        int ** nums_d = (int **) malloc ((nthreads - 1) * sizeof(int*));

        //hipSetDevice(0);
        hipMalloc(&num_d, sizeof(int));
        hipMemcpyAsync(num_d, num_h, sizeof(int), hipMemcpyHostToDevice, streams[0]);
        hipMemcpyAsync(num_1_d, num_1, sizeof(int), hipMemcpyHostToDevice, streams[0]);

        for(int i = 0; i < nthreads - 1; i++) {
                hipSetDevice(i+1);
                hipHostMalloc(&nums[i], sizeof(int));
                *nums[i] = 3;
                hipMalloc(&nums_d[i], sizeof(int));
        }

        hipSetDevice(0);
#pragma omp parallel num_threads(nthreads)
        {
                int tid = omp_get_thread_num();
                if(tid == 0) {
			clock_gettime(CLOCK_MONOTONIC, &start);	
			hipLaunchKernelGGL(kernelAdd, 1, 1, 0, streams[tid], 2, num_d);
			hipEventRecord(kernelEvent, streams[tid]);
        		hipLaunchKernelGGL(kernelMult, 1, 1, 0, streams[tid], num_d, num_1_d);
        		//hipMemcpyAsync(num_1, num_1_d, sizeof(int), hipMemcpyDeviceToHost, streams[tid]);	
                } else {
			hipSetDevice(tid);
			hipStreamWaitEvent(streams[tid], kernelEvent,0);
                	//hipMemcpyAsync(nums_d[tid-1], nums[tid-1], sizeof(int), hipMemcpyHostToDevice, streams[tid]);
                	hipLaunchKernelGGL(kernelMult, 1, 1, 0, streams[tid], num_d, nums_d[tid-1]);
                	//hipMemcpyAsync(nums[tid-1], nums_d[tid-1], sizeof(int), hipMemcpyDeviceToHost, streams[tid]);
                }
                hipStreamSynchronize(streams[tid]);
        }
        clock_gettime(CLOCK_MONOTONIC, &end); /* mark the end time */
        diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
        printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
        printf("num_1: %d\n", *num_1);
        for(int i = 0; i < nthreads - 1; i++)
                printf("num_%d: %d\n", i+2, *nums[i]);

        for(int i = 0; i < nthreads; i++) {
                hipSetDevice(i);
                hipStreamDestroy(streams[i]);
		if(i == 0) {
			hipEventDestroy(kernelEvent);	
		}
        }		

        for(int i = 0; i < nthreads - 1; i++) {
                hipSetDevice(i+1);
                hipHostFree(nums[i]);
                hipFree(nums_d[i]);
        }

	hipSetDevice(0);
        free(nums);
        free(nums_d);

        hipHostFree(num_h);
        hipFree(num_d);

        hipHostFree(num_1);
        hipFree(num_1_d);

        free(streams);
        //free(kernelEvent);
        return 0;
}
