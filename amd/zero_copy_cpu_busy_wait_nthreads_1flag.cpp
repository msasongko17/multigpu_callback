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
void kernelAdd(int inc, int * num, int * mult, int nthreads, int * cpu_flag_pointer)
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

                for(int j = 0; j < nthreads; j++) {
                        if(i != j) {
                                GUARD_CUDACALL2(hipDeviceEnablePeerAccess(j, 0), "hipDeviceEnablePeerAccess", __LINE__);
                        }
                }

                GUARD_CUDACALL2(hipStreamCreate(&(streams[i])), "hipStreamCreate", __LINE__);
        }


        GUARD_CUDACALL2(hipSetDevice(0), "hipSetDevice", __LINE__);
	int * cpu_flag;
        hipHostAlloc((void **)&cpu_flag, sizeof(int), hipHostMallocMapped);	

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

        int **cpu_flag_pointer = (int **) malloc (sizeof(int *));
	*cpu_flag = 0;	

        //hipSetDevice(0);
        hipMalloc(&num_d, sizeof(int));
        hipMemcpyAsync(num_d, num_h, sizeof(int), hipMemcpyHostToDevice, streams[0]);
        hipMemcpyAsync(num_1_d, num_1, sizeof(int), hipMemcpyHostToDevice, streams[0]);

        hipHostGetDevicePointer((void **)cpu_flag_pointer, (void *)cpu_flag, 0);
        for(int i = 0; i < nthreads - 1; i++) {
                hipSetDevice(i+1);
                hipHostMalloc(&nums[i], sizeof(int));
                *nums[i] = 3;
                hipMalloc(&nums_d[i], sizeof(int));
        }

	hipSetDevice(0);
        //clock_gettime(CLOCK_MONOTONIC, &start);
#pragma omp parallel num_threads(nthreads)
        {
                int tid = omp_get_thread_num();
                if(tid == 0) {
			clock_gettime(CLOCK_MONOTONIC, &start);
                        hipLaunchKernelGGL(kernelAdd, 1, 1, 0, streams[tid], 2, num_d, num_1_d, nthreads, *cpu_flag_pointer);
                        //hipMemcpyAsync(num_1, num_1_d, sizeof(int), hipMemcpyDeviceToHost, streams[tid]);
                        //hipEventRecord(kernelEvent[0], streams[0]);
                } else {
                        //fprintf(stderr, "before while in thread %d\n", tid);
			hipSetDevice(tid);
                        while(*cpu_flag == 0);
                        //fprintf(stderr, "after while in thread %d\n", tid);
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
        }

        hipSetDevice(0);

        for(int i = 0; i < nthreads - 1; i++) {
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
