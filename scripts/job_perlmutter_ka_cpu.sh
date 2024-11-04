#!/bin/bash
#SBATCH -A ntrain1
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -J gs-julia-1MPI-CPU_KA
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 0:02:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

#export SLURM_CPU_BIND="cores"
GS_DIR=$SCRATCH/GrayScott.jl
GS_EXE=$GS_DIR/gray-scott.jl

module load julia/1.10

export JULIA_NUM_THREADS=64

srun -n 1 --gpus=1 julia --project=$GS_DIR $GS_EXE $GS_DIR/test/functional/config_cpu_ka.toml
