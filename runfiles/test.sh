#!/bin/bash
#PBS -N test
#PBS -o /home/cgubeno/log/
#PBS -e /home/cgubeno/log/
#PBS -l select=2:ncpus=52:mem=375gb
#PBS -l walltime=10:00:00
#PBS -u cgubeno
#PBS -m ae
#PBS -q small
#PBS -k doe

source ~/.bashrc
conda activate megatop

which mpirun
mpirun --version

mpirun -np 2 hostname