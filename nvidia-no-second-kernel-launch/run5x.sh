#! /bin/bash

echo "cpu_initiated_1thread 1 thread" 2>&1 | tee 1thread_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 1 2>&1 | tee -a 1thread_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 1 2>&1 | tee -a 1thread_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 1 2>&1 | tee -a 1thread_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 1 2>&1 | tee -a 1thread_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 1 2>&1 | tee -a 1thread_log
echo "zero_copy_cpu_busy_wait_1thread 1 thread" 2>&1 | tee -a 1thread_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_1thread 1 2>&1 | tee -a 1thread_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_1thread 1 2>&1 | tee -a 1thread_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_1thread 1 2>&1 | tee -a 1thread_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_1thread 1 2>&1 | tee -a 1thread_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_1thread 1 2>&1 | tee -a 1thread_log
echo "cudastreamwaitevent_1thread 1 thread" 2>&1 | tee -a 1thread_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_1thread 1 2>&1 | tee -a 1thread_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_1thread 1 2>&1 | tee -a 1thread_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_1thread 1 2>&1 | tee -a 1thread_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_1thread 1 2>&1 | tee -a 1thread_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_1thread 1 2>&1 | tee -a 1thread_log
