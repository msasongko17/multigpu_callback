install:
	nvcc zero_copy_cpu_busy_wait_nthreads.cu -o zero_copy_cpu_busy_wait_nthreads -Xcompiler -fopenmp

run:
	./zero_copy_cpu_busy_wait_nthreads 2
	./zero_copy_cpu_busy_wait_nthreads 3
	./zero_copy_cpu_busy_wait_nthreads 4

clean: 
	rm zero_copy_cpu_busy_wait_nthreads	
