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
//#include <hip/hip_runtime.h>
#if defined(__x86_64__) || defined(__i386__) ||defined(__arm__)
#include <asm/perf_regs.h>
#endif

#include <time.h>

#define SET_MEM_OFFSET 103
#define SET_MEM_SIZE 104
#define CHECK_MEM 105
#define CHECK_VAR 106
// after

#define SAMPLE_VERSION "HIP-Examples-Application-v1.0"
#define SUCCESS 0
#define FAILURE 1

using namespace std;

__global__ void write_to_array(volatile uint32_t* var1)
{
#if 0
	array1[0] = 1;
	for(int i = 1; i < size; i++) {
		array1[i] = i;
	}
#endif
	*var1 = 13;	

	//__threadfence();
}

//uint32_t var1;

int main(int argc, char* argv[])
{
#if 0
	int fd1;
	fd1 = open("/dev/kfd", O_RDWR);

	if(fd1 < 0) {
		printf("cannot open the device file...\n");
		return 0;
	}
#endif 
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	cout << " System minor " << devProp.minor << endl;
	cout << " System major " << devProp.major << endl;
	cout << " agent prop name " << devProp.name << endl;

	/* Initial input,output for the host and create memory objects for the kernel*/
	uint32_t size = 14;
	//uint32_t *array1 = NULL;
	uint32_t *var1;
	//uint32_t var1;

	cudaHostAlloc((void**)&var1, sizeof(uint32_t), cudaHostAllocMapped );
	//ioctl(fd1, SET_MEM_OFFSET, var1);
        //ioctl(fd1, SET_MEM_SIZE, size);

	uint32_t *dev_var;	

	cudaHostGetDevicePointer((void **) &dev_var, (void *) var1, 0);
#if 0
	hipLaunchKernelGGL(write_to_array,
                  dim3(1),
                  dim3(1),
                  0, 0,
                  dev_var);
#endif
	write_to_array<<<1, 1, 0>>>(dev_var);	
//#if 0
	//for(int i = 0; i < 100000000; i++);
	//sleep((long double) 0.00000001);
	while(*var1 == 0);
	cout << "var1: " << *var1 << endl;
	//sleep((long double) 0.00000001);
	//ioctl(fd1, CHECK_VAR);
        //ioctl(fd1, FREE_MEM);
        //close(fd1);
//#endif	
#if 0
	cout << "\noutput: " << endl;
	for(int i = 0; i < size; i++) {
		cout << array1[i] << " ";
	}
	cout << endl;
#endif
#if 0
	ioctl(fd1, UPDATE_MEM);
        ioctl(fd1, FREE_MEM);
        close(fd1);
#endif
	cudaFree(var1);
	std::cout<<"Passed!\n";
	return SUCCESS;
}
