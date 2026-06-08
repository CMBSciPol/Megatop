
#!/bin/bash
#PBS -N snake_planck
#PBS -o /home/cgubeno/log/
#PBS -e /home/cgubeno/log/
#PBS -l select=1:ncpus=52:mem=375gb
#PBS -l walltime=700:00:00
#PBS -u cgubeno
#PBS -m ae
#PBS -q small
#PBS -k doe

set -euo pipefail

# run from submit directory
cd "$PBS_O_WORKDIR"

ENV_BIN="$HOME/.conda/envs/megatop/bin"
export PATH="$ENV_BIN:$PATH"
hash -r

PARAM_FILE="/home/cgubeno/Megatop/paramfiles/planck_npipe_masked_1024.yaml"

echo "Running pipeline with paramfile: ${PARAM_FILE}"
echo ""

snakemake --cores 25 --configfile "$PARAM_FILE" 


echo "Plotting Outputs"

mpirun -n 1 megatop-mask-plot --config ${PARAM_FILE}

# mpirun -n 1 megatop-mock-plot --config ${PARAM_FILE}

mpirun -n 1 megatop-preproc-plot --config ${PARAM_FILE}
mpirun -n 1 megatop-noisecov-plot --config ${PARAM_FILE}
mpirun -n 1 megatop-compsep-plot --config ${PARAM_FILE}
mpirun -n 1 megatop-map2cl-plot --config ${PARAM_FILE}
mpirun -n 1 megatop-noisespectra-plot --config ${PARAM_FILE}
mpirun -n 1 megatop-cl2r-plot --config ${PARAM_FILE}
mpirun -n 1 megatop-cl2r_mcmc-plot --config ${PARAM_FILE}