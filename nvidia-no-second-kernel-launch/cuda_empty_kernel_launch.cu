#include <stdio.h>

__global__ void EmptyKernel() { }

int main() {

    const int N = 1000;

    float time, cumulative_time = 0.f;
    float milliseconds = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i=0; i<N; i++) { 

        cudaEventRecord(start, 0);
        EmptyKernel<<<1,1>>>(); 
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        //cudaEventElapsedTime(&time, start, stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
        cumulative_time = cumulative_time + milliseconds;

    }

    //float milliseconds; //= (float *) malloc(ngpus * sizeof(float));

        //for(int i = 0; i < ngpus; i++) {
        //cudaEventElapsedTime(&milliseconds, eventStart[0], eventStop[0]);
    //printf("elapsed time in gpu 0 = %0.2f nanoseconds\n", milliseconds * 1000000);
    printf("Kernel launch overhead time:  %3.5f microseconds \n", cumulative_time / N * 1000);
    return 0;
}
