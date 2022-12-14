#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdint.h>     /* for uint64 definition */
#include <stdlib.h>     /* for exit() definition */
#include <time.h>       /* for clock_gettime */
#include <omp.h>

__global__ void EmptyKernel() { }

int main() {

    const int N = 5;

    float time, cumulative_time = 0.f;
    float milliseconds = 0.f;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    for (int i=0; i<N; i++) { 

        hipEventRecord(start, 0);
        //EmptyKernel<<<1,1>>>(); 
	hipLaunchKernelGGL(EmptyKernel, dim3(1), dim3(1), 0, 0);
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        //cudaEventElapsedTime(&time, start, stop);
	hipEventElapsedTime(&milliseconds, start, stop);
        cumulative_time = cumulative_time + milliseconds;
	printf("Kernel launch overhead time:  %3.5f microseconds \n", milliseconds * 1000);
    }

    //float milliseconds; //= (float *) malloc(ngpus * sizeof(float));

        //for(int i = 0; i < ngpus; i++) {
        //cudaEventElapsedTime(&milliseconds, eventStart[0], eventStop[0]);
    //printf("elapsed time in gpu 0 = %0.2f nanoseconds\n", milliseconds * 1000000);
    printf("Kernel launch overhead time:  %3.5f microseconds \n", cumulative_time / N * 1000);
    return 0;
}
