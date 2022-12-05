/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
// before
#define _GNU_SOURCE 1

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <string>
#include <fstream>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#include <errno.h>

#include <signal.h>
#include <sys/mman.h>

#include <sys/ioctl.h>
#include <asm/unistd.h>
#include <sys/prctl.h>
#include <hip/hip_runtime.h>
#if defined(__x86_64__) || defined(__i386__) ||defined(__arm__)
#include <asm/perf_regs.h>
#endif

#include <time.h>

#define SET_MEM_OFFSET 103
#define SET_MEM_SIZE 104
#define UPDATE_MEM 105
#define FREE_MEM 106
#define REGISTER_SIGNAL_RECIPIENT 107
#define SET_USER_PAGE_OFFSET 108
#define UPDATE_USER_MEM 109
#define FREE_USER_MEM 110
#define CHECK_USER_MEM 111
#define SET_USER_MEM_OFFSET 112
// after

#define SAMPLE_VERSION "HIP-Examples-Application-v1.0"
#define SUCCESS 0
#define FAILURE 1
#define SIGNEW 44
#define BILLION 1000000000L

using namespace std;
hipStream_t *streams;
int ngpus;
uint64_t *ptr;

struct timespec start_time, end_time;
uint64_t diff_time;

__global__ void receive_interrupt(uint64_t gpu_id)
{
        //printf("GPU: %d is running in receive_interrupt\n", gpu_id);
}

void kernel_launch(uint32_t target_gpu) {
        hipSetDevice(target_gpu);
	//cout << "receive_interrupt will be called\n";
#if 0
	hipLaunchKernelGGL(receive_interrupt,
                  dim3(1),
                  dim3(1),
                  0, streams[target_gpu],
                  target_gpu);	
#endif
	//cout << "receive_interrupt is called\n";
        hipStreamSynchronize(streams[target_gpu]);
	clock_gettime(CLOCK_MONOTONIC, &end_time); /* mark the end time */
        diff_time = BILLION * (end_time.tv_sec - start_time.tv_sec) + end_time.tv_nsec - start_time.tv_nsec;
        printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff_time);
	hipStreamDestroy(streams[target_gpu]);
}

#if 0
void sig_event_handler(int n, siginfo_t *info, void *unused)
{
        if (n == SIGNEW) {
		ptr[0] = 0;
		//cout << "sig_event_handler is called 1\n";
		for (int i = 0; i < ngpus; i++)
                	kernel_launch(i);
		//cout << "sig_event_handler is called 2\n";
        }
}
#endif

void sig_event_handler(int n, siginfo_t *info, void *unused)
{
        if (n == SIGNEW) {
                ptr[0] = 0;
                //cout << "sig_event_handler is called 1\n";
		for(int i = 0; i < ngpus; i++) {
                	hipSetDevice(i);
                	hipStreamSynchronize(streams[i]);
        	}
		clock_gettime(CLOCK_MONOTONIC, &end_time); /* mark the end time */
        	diff_time = BILLION * (end_time.tv_sec - start_time.tv_sec) + end_time.tv_nsec - start_time.tv_nsec;
        	printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff_time);
		for(int i = 0; i < ngpus; i++) {
                        hipSetDevice(i);
                        hipStreamDestroy(streams[i]);
                }
                //cout << "sig_event_handler is called 2\n";
        }
}

__global__ void send_interrupt(volatile uint64_t* array1, uint32_t size)
{
#if 0
	array1[0] = 1;
//#if 0
        for(int i = 1; i < size; i++) {
                array1[i] = i;
        }
#endif	
	array1[0] = 1;
#if 0
	for(int i = 1; i < 10; i++) {
                array1[i] = i;
        }
#endif
	asm volatile ("s_sendmsg 0x1;");
}

int main(int argc, char* argv[])
{
// before
	if(argc != 2) {
                fprintf(stderr, "usage: ./bench <thread-count>\n");
                return -1;
        }
        ngpus = atoi(argv[1]);
        if(ngpus <= 0) {
                fprintf(stderr, "GPU count is supposed to be a nonzero integer\n");
                return -1;
        }	
//after
	int fd1;
	fd1 = open("/dev/kfd", O_RDWR);

	if(fd1 < 0) {
		printf("cannot open the device file...\n");
		return 0;
	} 
	hipDeviceProp_t devProp;
	hipGetDeviceProperties(&devProp, 0);
	cout << " System minor " << devProp.minor << endl;
	cout << " System major " << devProp.major << endl;
	cout << " agent prop name " << devProp.name << endl;	

	int num_gpus = 0;
	hipGetDeviceCount (&num_gpus);

	if(ngpus > num_gpus) {
                fprintf(stderr, "GPU count cannot be higher than the number of available GPUs.\n");
                return -1;
        }

	streams = (hipStream_t *) malloc(ngpus * sizeof(hipStream_t)); 

	struct sigaction act;
        sigemptyset(&act.sa_mask);
        act.sa_flags = (SA_SIGINFO | SA_RESTART);
        act.sa_sigaction = sig_event_handler;
        sigaction(SIGNEW, &act, NULL);	

	for(int i = 0; i < ngpus; i++) {
                hipSetDevice(i);
                hipSetDeviceFlags(hipDeviceScheduleBlockingSync);
                hipStreamCreate(&(streams[i]));
                ioctl(fd1, REGISTER_SIGNAL_RECIPIENT, i);
        }	

	uint32_t size = 10;
	uint64_t *dev_ptr = NULL;
        //posix_memalign((void **)&ptr, 4096, size * sizeof(uint64_t));
	hipHostMalloc((void**)&ptr, size * sizeof(uint64_t), hipHostMallocDefault );
	uint64_t *offset = (uint64_t *) ((uint64_t) ptr & (~ ((uint64_t) 0x000fff)));
	fprintf(stderr, "offset address of the page is %lx, the allocated address is: %lx\n", (unsigned long) offset, (unsigned long) ptr);
	ioctl(fd1, SET_USER_PAGE_OFFSET, offset);
	ioctl(fd1, SET_MEM_SIZE, size);
	ioctl(fd1, SET_USER_MEM_OFFSET, (uint64_t) ptr - (uint64_t) offset);
	//ioctl(fd1, UPDATE_USER_MEM);
#if 0
	for(int i = 1; i < 10;) {
                ptr[i] = i;
                i = i + 2;
        }
#endif
	//ptr = (uint64_t *) malloc (size * sizeof(uint64_t));
        //ioctl(fd1, SET_USER_MEM_OFFSET, ptr);
//#if 0
	hipHostGetDevicePointer((void **) &dev_ptr, (void *) ptr, 0);
        //ioctl(fd1, UPDATE_USER_MEM);	

	hipSetDevice(0);
	clock_gettime(CLOCK_MONOTONIC, &start_time);
	hipLaunchKernelGGL(send_interrupt,
                  dim3(1),
                  dim3(1),
                  0, streams[0],
		  dev_ptr, size);

	//for(int i = 0; i < ngpus; i++) {
		
        //hipStreamSynchronize(streams[0]);
	//hipStreamDestroy(streams[0]);

        ioctl(fd1, FREE_USER_MEM);
        close(fd1);	
        //ioctl(fd1, FREE_USER_MEM);
        hipFree(ptr);
	std::cout<<"Passed!\n";
	return SUCCESS;
}
