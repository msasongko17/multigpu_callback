#! /bin/bash

echo "cpu_initiated_1thread 1 GPUs"
./cpu_initiated_1thread 1
./cpu_initiated_1thread 1
./cpu_initiated_1thread 1
./cpu_initiated_1thread 1
./cpu_initiated_1thread 1
echo "hipstreamwaitevent_1thread 1 GPUs"
./hipstreamwaitevent_1thread 1
./hipstreamwaitevent_1thread 1
./hipstreamwaitevent_1thread 1
./hipstreamwaitevent_1thread 1
./hipstreamwaitevent_1thread 1
echo "zero_copy_cpu_busy_wait_1thread 1 GPUs"
./zero_copy_cpu_busy_wait_1thread 1
./zero_copy_cpu_busy_wait_1thread 1
./zero_copy_cpu_busy_wait_1thread 1
./zero_copy_cpu_busy_wait_1thread 1
./zero_copy_cpu_busy_wait_1thread 1
