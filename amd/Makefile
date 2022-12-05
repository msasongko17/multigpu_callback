install:
	hipcc zero_copy_cpu_busy_wait_1thread.cpp -o zero_copy_cpu_busy_wait_1thread
	hipcc hipstreamwaitevent_1thread.cpp -o hipstreamwaitevent_1thread
	hipcc cpu_initiated_1thread.cpp -o cpu_initiated_1thread

run:
	GOMP_CPU_AFFINITY="0 1 2 3" ./zero_copy_cpu_busy_wait_1thread 2  2>&1 | tee 1_threads_2_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3" ./hipstreamwaitevent_1thread 2 2>&1 | tee 1_threads_2_gpus_hipstreamwaitevent_log 
	
run2:
	GOMP_CPU_AFFINITY="0 1 2 3" ./zero_copy_cpu_busy_wait_1thread 2  2>&1 | tee 1_threads_2_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3" ./hipstreamwaitevent_1thread 2 2>&1 | tee 1_threads_2_gpus_hipstreamwaitevent_log

run3:
	GOMP_CPU_AFFINITY="0 1 2 3" ./zero_copy_cpu_busy_wait_1thread 2  2>&1 | tee 1_threads_2_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3" ./hipstreamwaitevent_1thread 2 2>&1 | tee 1_threads_2_gpus_hipstreamwaitevent_log

run4:
	GOMP_CPU_AFFINITY="0 1 2 3" ./zero_copy_cpu_busy_wait_1thread 2  2>&1 | tee 1_threads_2_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3" ./hipstreamwaitevent_1thread 2 2>&1 | tee 1_threads_2_gpus_hipstreamwaitevent_log

run5:
	GOMP_CPU_AFFINITY="0 1 2 3" ./zero_copy_cpu_busy_wait_1thread 2  2>&1 | tee 1_threads_2_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3" ./hipstreamwaitevent_1thread 2 2>&1 | tee 1_threads_2_gpus_hipstreamwaitevent_log

result1:
	mkdir results1 && mv *log results1

result2:
	mkdir results2 && mv *log results2

result3:
	mkdir results3 && mv *log results3

result4:
	mkdir results4 && mv *log results4

result5:
	mkdir results5 && mv *log results5

scalability:	install run result1 run2 result2 run3 result3 run4 result4 run5 result5

clean: 
	rm zero_copy_cpu_busy_wait_1thread hipstreamwaitevent_1thread cpu_initiated_1thread

