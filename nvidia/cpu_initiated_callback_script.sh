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
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 2  2>&1 | tee cpu_initiated_1_threads_2_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 4 2>&1 | tee cpu_initiated_1_threads_4_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 8 2>&1 | tee cpu_initiated_1_threads_8_gpus_log
mkdir results1
mv *log results1
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 2  2>&1 | tee cpu_initiated_1_threads_2_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 4 2>&1 | tee cpu_initiated_1_threads_4_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 8 2>&1 | tee cpu_initiated_1_threads_8_gpus_log
mkdir results2
mv *log results2
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 2  2>&1 | tee cpu_initiated_1_threads_2_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 4 2>&1 | tee cpu_initiated_1_threads_4_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 8 2>&1 | tee cpu_initiated_1_threads_8_gpus_log
mkdir results3
mv *log results3
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 2  2>&1 | tee cpu_initiated_1_threads_2_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 4 2>&1 | tee cpu_initiated_1_threads_4_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 8 2>&1 | tee cpu_initiated_1_threads_8_gpus_log
mkdir results4
mv *log results4
GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 2  2>&1 | tee cpu_initiated_1_threads_2_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 4 2>&1 | tee cpu_initiated_1_threads_4_gpus_log && GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7" ./cpu_initiated_1thread 8 2>&1 | tee cpu_initiated_1_threads_8_gpus_log
mkdir results5
mv *log results5
