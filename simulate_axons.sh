#!/bin/bash

#SBATCH --job-name=simulate_axon         # Job name
#SBATCH --output=/mnt/scratch/personal/jesperdn/slurm_logs/%x_%A_%a.log          # A = master job id, a = task job id
#SBATCH --nodes=1                   # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --ntasks=1                  # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --cpus-per-task=1          # Relevant when program implements MP (single system parallelism, e.g., OpenMP, TBB)
#SBATCH --mem=2G                   # Job memory request
#SBATCH --array=0-3965           # or 1,2,4,5,9 ; access as $SLURM_ARRAY_TASK_ID 5279
#SBATCH --exclude=g[17,51]

# NO USE: SBATCH --nodelist=bigger[4,6,8],fenrir,chimera,gojira

echo "Job Information"
echo
echo "Job name     :  $SLURM_JOB_NAME"
echo "Job ID       :  $SLURM_ARRAY_JOB_ID"
echo "Task ID      :  $SLURM_ARRAY_TASK_ID"
echo "Cluster name :  $SLURM_CLUSTER_NAME"
echo "Node name    :  $SLURM_NODENAME"
echo "Node list    :  $SLURM_NODELIST"
echo "Date         :  $(date)"
echo "Working dir  :  $SLURM_SUBMIT_DIR"
echo


source ~/mambaforge/etc/profile.d/conda.sh
conda activate neurosimnibs

export NEURON_MODULE_OPTIONS="-nogui"

cd /home/jesperdn/repositories/project-long_range_axon_projections

echo Running simulations
# python torge_script/simulate_axon.py 5 $SLURM_ARRAY_TASK_ID pa monophasic smooth
# python torge_script/simulate_axon.py 5 $SLURM_ARRAY_TASK_ID ap monophasic smooth

python torge_script/simulate_axon.py 5 $SLURM_ARRAY_TASK_ID pa biphasic smooth
python torge_script/simulate_axon.py 5 $SLURM_ARRAY_TASK_ID ap biphasic smooth

# # time constant: diameter = 6.0
# python torge_script/simulate_axon.py 5 $SLURM_ARRAY_TASK_ID pa C_TMS_30 smooth
# python torge_script/simulate_axon.py 5 $SLURM_ARRAY_TASK_ID pa C_TMS_60 smooth
# python torge_script/simulate_axon.py 5 $SLURM_ARRAY_TASK_ID pa C_TMS_120 smooth

# # default cond: diameter = 2.0
# python torge_script/simulate_axon.py 5 $SLURM_ARRAY_TASK_ID ap monophasic
# python torge_script/simulate_axon.py 5 $SLURM_ARRAY_TASK_ID pa monophasic
