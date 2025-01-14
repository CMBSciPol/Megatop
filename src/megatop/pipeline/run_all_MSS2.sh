#!/bin/bash
#SBATCH -N 2
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -J MSS2_NoiseCov
#SBATCH --mail-user=baptiste.jost@ipmu.jp
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


paramfile='/global/homes/j/jost/Megatop/paramfiles/MSS2.yml'

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


echo "------------------------------------------------------------"
echo "|                       PRE-PROCESSING                     |"
echo "------------------------------------------------------------"
python pre_processing.py --globals ${paramfile} --plots


echo "------------------------------------------------------------"
echo "|                NOISE-COVARIANCE COMPUTATION              |"
echo "------------------------------------------------------------"
# Creating noise sims / preprocessing them and computing noise covariance
srun -n 50 -c 10 --mpi=pmi2 --cpu_bind=cores python get_noise_cov_pixel.py --globals ${paramfile} --sims True --use_mpi --verbose
# Computing noise covariance using the one (true) noise map per frequency from MSS2
python get_noise_cov_pixel.py --globals ${paramfile} --verbose --plots