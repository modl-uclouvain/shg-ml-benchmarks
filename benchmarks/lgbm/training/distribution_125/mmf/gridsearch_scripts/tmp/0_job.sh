#!/bin/bash
#
#SBATCH --job-name=0_LGBM
#
#SBATCH --partition=keira
#SBATCH --qos=keira
#SBATCH --nodes=1
#   #SBATCH --exclusive
#   #SBATCH	--exclude=mb-rom[203-206]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=5-00:00:00
#SBATCH --mem-per-cpu=1800
#SBATCH --output=./0_job.log
#SBATCH --error=./0_job.log
#

module --force purge
module restore Abi_dev

export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/venvs/modnenv_v2/bin/activate

# ulimit -s unlimited

cd /globalscratch/ucl/modl/vtrinque/NLO/shg_ml_benchmarks/lgbm/distribution_125/mmf/gridsearch_scripts

python 0_train_gdsearch.py >> ./0_job.log
echo "DONE"
