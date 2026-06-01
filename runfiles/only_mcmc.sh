
#!/bin/bash
#PBS -N mcmc_planck
#PBS -o /home/cgubeno/log/
#PBS -e /home/cgubeno/log/
#PBS -l select=1:ncpus=52:mem=150gb
#PBS -l walltime=10:00:00
#PBS -u cgubeno
#PBS -m ae
#PBS -q small
#PBS -k doe

set -euo pipefail

# activate python environment
cd "$PBS_O_WORKDIR"

ENV_BIN="$HOME/.conda/envs/megatop/bin"
export PATH="$ENV_BIN:$PATH"
hash -r

PARAM_FILE="/home/cgubeno/Megatop/paramfiles/planck_npipe_masked_bir_test.yaml"

echo "Running pipeline with paramfile: ${PARAM_FILE}"

echo "------------------------------------------------------------"
echo "|            COSMOLOGICAL PARAMETERS ESTIMATION            |"
echo "------------------------------------------------------------"
mpirun -n 50 megatop-cl2r-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting r statistics"
mpirun -n 1 megatop-cl2r-plot --config ${PARAM_FILE}
echo ""
echo "Plotting mcmc results statistics"
mpirun -n 1 megatop-cl2r_mcmc-plot --config ${PARAM_FILE}