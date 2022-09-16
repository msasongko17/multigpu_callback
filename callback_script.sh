#! /bin/bash
#SBATCH -p palamut-cuda 			 # Partition Name
#SBATCH -J callback 		 # Job Name
#SBATCH -N 1 				 # Number of Nodes
#SBATCH -n 1 				 # Number of tasks per Node
#SBATCH -c 128                             # Number of cores per Node
#SBATCH --gres=gpu:8 		 # GPU resources (can request specific as gpu:tesla_v100:1)
#SBATCH --time=00:10:00		 # Time day-hours:minutes:seconds
#SBATCH --output=callback.out	 # Standard output and error log to this file
lscpu
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_nthreads 2  2>&1 | tee 2_threads_2_flags_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_nthreads 4  2>&1 | tee 4_threads_4_flags_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_nthreads 8  2>&1 | tee 8_threads_8_flags_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_nthreads_1flag 2  2>&1 | tee 2_threads_1_flags_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_nthreads_1flag 4  2>&1 | tee 4_threads_1_flags_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_nthreads_1flag 8  2>&1 | tee 8_threads_1_flags_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_1thread 2  2>&1 | tee 1_threads_2_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_1thread 4  2>&1 | tee 1_threads_4_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_1thread 8  2>&1 | tee 1_threads_8_gpus_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_nthreads 2 2>&1 | tee 2_threads_2_flags_cudastreamwaitevent_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_nthreads 4 2>&1 | tee 4_threads_4_flags_cudastreamwaitevent_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_nthreads 8 2>&1 | tee 8_threads_8_flags_cudastreamwaitevent_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_nthreads_1flag 2 2>&1 | tee 2_threads_1_flags_cudastreamwaitevent_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_nthreads_1flag 4 2>&1 | tee 4_threads_1_flags_cudastreamwaitevent_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_nthreads_1flag 8 2>&1 | tee 8_threads_1_flags_cudastreamwaitevent_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_1thread 2 2>&1 | tee 1_threads_2_gpus_cudastreamwaitevent_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_1thread 4 2>&1 | tee 1_threads_4_gpus_cudastreamwaitevent_log  && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_1thread 8 2>&1 | tee 1_threads_8_gpus_cudastreamwaitevent_log
mkdir results1
mv *log results1
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_nthreads 2  2>&1 | tee 2_threads_2_flags_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_nthreads 4  2>&1 | tee 4_threads_4_flags_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_nthreads 8  2>&1 | tee 8_threads_8_flags_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_nthreads_1flag 2  2>&1 | tee 2_threads_1_flags_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_nthreads_1flag 4  2>&1 | tee 4_threads_1_flags_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_nthreads_1flag 8  2>&1 | tee 8_threads_1_flags_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_1thread 2  2>&1 | tee 1_threads_2_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_1thread 4  2>&1 | tee 1_threads_4_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_1thread 8  2>&1 | tee 1_threads_8_gpus_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_nthreads 2 2>&1 | tee 2_threads_2_flags_cudastreamwaitevent_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_nthreads 4 2>&1 | tee 4_threads_4_flags_cudastreamwaitevent_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_nthreads 8 2>&1 | tee 8_threads_8_flags_cudastreamwaitevent_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_nthreads_1flag 2 2>&1 | tee 2_threads_1_flags_cudastreamwaitevent_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_nthreads_1flag 4 2>&1 | tee 4_threads_1_flags_cudastreamwaitevent_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_nthreads_1flag 8 2>&1 | tee 8_threads_1_flags_cudastreamwaitevent_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_1thread 2 2>&1 | tee 1_threads_2_gpus_cudastreamwaitevent_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_1thread 4 2>&1 | tee 1_threads_4_gpus_cudastreamwaitevent_log  && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_1thread 8 2>&1 | tee 1_threads_8_gpus_cudastreamwaitevent_log
mkdir results2
mv *log results2
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_nthreads 2  2>&1 | tee 2_threads_2_flags_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_nthreads 4  2>&1 | tee 4_threads_4_flags_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_nthreads 8  2>&1 | tee 8_threads_8_flags_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_nthreads_1flag 2  2>&1 | tee 2_threads_1_flags_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_nthreads_1flag 4  2>&1 | tee 4_threads_1_flags_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_nthreads_1flag 8  2>&1 | tee 8_threads_1_flags_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_1thread 2  2>&1 | tee 1_threads_2_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_1thread 4  2>&1 | tee 1_threads_4_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./zero_copy_cpu_busy_wait_1thread 8  2>&1 | tee 1_threads_8_gpus_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_nthreads 2 2>&1 | tee 2_threads_2_flags_cudastreamwaitevent_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_nthreads 4 2>&1 | tee 4_threads_4_flags_cudastreamwaitevent_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_nthreads 8 2>&1 | tee 8_threads_8_flags_cudastreamwaitevent_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_nthreads_1flag 2 2>&1 | tee 2_threads_1_flags_cudastreamwaitevent_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_nthreads_1flag 4 2>&1 | tee 4_threads_1_flags_cudastreamwaitevent_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_nthreads_1flag 8 2>&1 | tee 8_threads_1_flags_cudastreamwaitevent_log
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_1thread 2 2>&1 | tee 1_threads_2_gpus_cudastreamwaitevent_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_1thread 4 2>&1 | tee 1_threads_4_gpus_cudastreamwaitevent_log  && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cudastreamwaitevent_1thread 8 2>&1 | tee 1_threads_8_gpus_cudastreamwaitevent_log
mkdir results3
mv *log results3

