#!/bin/bash
#
# all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
# set a job name
#SBATCH --job-name=dataset_generation
#################
# working directory
#SBATCH -D /home/asuso/trans_bezier
##############
# a file for job output, you can check job progress
#SBATCH --output=./ProbabilisticBezierEncoder/MultiBezierModels/ParallelVersion/dataset_generation.out
#################
# a file for errors from the job
#SBATCH --error=./ProbabilisticBezierEncoder/MultiBezierModels/ParallelVersion/dataset_generation.err
#################
# time you think you need
# In this case, hh:mm:ss, select whatever time you want, the less you ask for the # faster your job will run.
#SBATCH --time=50-00:00:00
#################
# --gres will give you one GPU, you can ask for more, up to 4 (or how ever many are on the node/card)
# 1080Ti, TitanXp
#SBATCH --gres gpu:1
# We are submitting to the batch partition
#SBATCH -p dag
#################
# Number of cores
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
# Ensure that all cores are on one machine
#SBATCH -N 1
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=6000
#################
#SBATCH --export=ALL

#################
# Input variable indicating the experiment to run
# set -- $1

#################
# Load specific experiment variable
# . scripts/experiments.sh

#################
# Prepare the experiment to run
CODE="python -u ProbabilisticBezierEncoder/MultiBezierModels/FixedCP/dataset_generation.py"

#################
# Prepare the experiment to run
# echo "Experiment number: $1"
echo $CODE

#################
# Run the experiment
srun $CODE

#################
# Report that it has finished
echo "Done."

#free -h
