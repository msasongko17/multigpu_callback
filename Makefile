install:
	nvcc zero_copy_cpu_busy_wait_nthreads.cu -o zero_copy_cpu_busy_wait_nthreads -Xcompiler -fopenmp
	nvcc zero_copy_cpu_busy_wait_nthreads_1flag.cu -o zero_copy_cpu_busy_wait_nthreads_1flag -Xcompiler -fopenmp
	nvcc zero_copy_cpu_busy_wait_1thread.cu -o zero_copy_cpu_busy_wait_1thread
	nvcc cudastreamwaitevent_nthreads.cu -o cudastreamwaitevent_nthreads -Xcompiler -fopenmp
	nvcc cudastreamwaitevent_nthreads_1flag.cu -o cudastreamwaitevent_nthreads_1flag -Xcompiler -fopenmp
	nvcc cudastreamwaitevent_1thread.cu -o cudastreamwaitevent_1thread

run:
	./zero_copy_cpu_busy_wait_nthreads 2  2>&1 | tee 2_threads_2_flags_log && ./zero_copy_cpu_busy_wait_nthreads 3  2>&1 | tee 3_threads_3_flags_log && ./zero_copy_cpu_busy_wait_nthreads 4  2>&1 | tee 4_threads_4_flags_log 
	./zero_copy_cpu_busy_wait_nthreads_1flag 2  2>&1 | tee 2_threads_1_flags_log && ./zero_copy_cpu_busy_wait_nthreads_1flag 3  2>&1 | tee 3_threads_1_flags_log && ./zero_copy_cpu_busy_wait_nthreads_1flag 4  2>&1 | tee 4_threads_1_flags_log
	./zero_copy_cpu_busy_wait_1thread 2  2>&1 | tee 1_threads_2_gpus_log && ./zero_copy_cpu_busy_wait_1thread 3  2>&1 | tee 1_threads_3_gpus_log && ./zero_copy_cpu_busy_wait_1thread 4  2>&1 | tee 1_threads_4_gpus_log
	./cudastreamwaitevent_nthreads 2 2>&1 | tee 2_threads_2_flags_cudastreamwaitevent_log && ./cudastreamwaitevent_nthreads 3 2>&1 | tee 3_threads_3_flags_cudastreamwaitevent_log && ./cudastreamwaitevent_nthreads 4 2>&1 | tee 4_threads_4_flags_cudastreamwaitevent_log
	./cudastreamwaitevent_nthreads_1flag 2 2>&1 | tee 2_threads_1_flags_cudastreamwaitevent_log && ./cudastreamwaitevent_nthreads_1flag 3 2>&1 | tee 3_threads_1_flags_cudastreamwaitevent_log && ./cudastreamwaitevent_nthreads_1flag 4 2>&1 | tee 4_threads_1_flags_cudastreamwaitevent_log
	./cudastreamwaitevent_1thread 2 2>&1 | tee 1_threads_2_gpus_cudastreamwaitevent_log && ./cudastreamwaitevent_1thread 3 2>&1 | tee 1_threads_3_gpus_cudastreamwaitevent_log  && ./cudastreamwaitevent_1thread 4 2>&1 | tee 1_threads_4_gpus_cudastreamwaitevent_log 

result1:
	mkdir results1 && mv *log results1

result2:
	mkdir results2 && mv *log results2

result2:
	mkdir results3 && mv *log results3

scalability:	install run result1 run result2 run result3

clean: 
	rm zero_copy_cpu_busy_wait_nthreads zero_copy_cpu_busy_wait_nthreads_1flag zero_copy_cpu_busy_wait_1thread cudastreamwaitevent_nthreads cudastreamwaitevent_nthreads_1flag cudastreamwaitevent_1thread

