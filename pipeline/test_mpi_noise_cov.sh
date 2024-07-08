#!/bin/bash
#SBATCH -N 2
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -J 250NoiseCov
#SBATCH --mail-user=baptiste.jost@ipmu.jp
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


paramfile='/global/homes/j/jost/Megatop/paramfiles/test_preproc_mpi250.yml'
simparamfile='/global/homes/j/jost/Megatop/paramfiles/simulations250.yml'

echo "Running pipeline with paramfile: ${paramfile}"

echo "------------------------------------------------------------"
echo "|           PREPARING METADATA AND SIMULATIONS             |"
echo "------------------------------------------------------------"

echo "Pre-processing data..."
echo "-------------------"

conda init bash
source ~/.bashrc
conda activate megatop


echo "------------------------------------------------------------"
echo "|                       MASK-HANDLING                      |"
echo "------------------------------------------------------------"
python mask_handler.py --globals ${paramfile} --plots --verbose
python mask_handler.py --globals ${simparamfile} --plots --verbose


echo "------------------------------------------------------------"
echo "|                       PRE-PROCESSING                     |"
echo "------------------------------------------------------------"
srun -n 250 -c 2 --mpi=pmi2 --cpu_bind=cores python pre_processing.py --globals ${paramfile} --sims ${simparamfile} --plots --use_mpi


echo "------------------------------------------------------------"
echo "|                NOISE-COVARIANCE COMPUTATION              |"
echo "------------------------------------------------------------"
srun -n 250 -c 2 --mpi=pmi2 --cpu_bind=cores python get_noise_cov_pixel.py --globals ${paramfile} --sims ${simparamfile} --plots --use_mpi