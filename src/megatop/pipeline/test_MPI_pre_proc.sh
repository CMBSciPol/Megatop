#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -J MPI_PreProc_test
#SBATCH --mail-user=baptiste.jost@ipmu.jp
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


paramfile='/global/homes/j/jost/Megatop/paramfiles/test_preproc_mpi.yml'
simparamfile='/global/homes/j/jost/Megatop/paramfiles/simulations.yml'

echo "Running pipeline with paramfile: ${paramfile}"

echo "------------------------------------------------------------"
echo "|           PREPARING METADATA AND SIMULATIONS             |"
echo "------------------------------------------------------------"

echo "Pre-processing data..."
echo "-------------------"

conda init bash
source ~/.bashrc
conda activate megatop

python mask_handler.py --globals ${paramfile} --plots --verbose
python mask_handler.py --globals ${simparamfile} --plots --verbose
srun -n 10 -c 24 --mpi=pmi2 --cpu_bind=cores python pre_processing.py --globals ${paramfile} --sims ${simparamfile} --plots --use_mpi
