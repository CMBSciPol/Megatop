
#!/bin/bash
#PBS -N planck_megatop
#PBS -o /home/cgubeno/log/
#PBS -e /home/cgubeno/log/
#PBS -l select=1:ncpus=20:mem=375gb
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

PARAM_FILE="/home/cgubeno/Megatop/paramfiles/planck_npipe_masked_bir.yaml"

echo "Running pipeline with paramfile: ${PARAM_FILE}"

echo ""
echo ""
echo "Plotting mask outputs"
mpirun -n 1 megatop-mask-plot --config ${PARAM_FILE}


# echo "Plotting mocker outputs"
# mpirun -n 1 megatop-mock-plot --config ${PARAM_FILE}



echo ""
echo ""
echo "Plotting pre-processer outputs"
mpirun -n 1 megatop-preproc-plot --config ${PARAM_FILE}


echo ""
echo ""
echo "Plotting noise covariance outputs"
mpirun -n 1 megatop-noisecov-plot --config ${PARAM_FILE}

echo ""
echo ""
echo "Plotting component separater outputs"
mpirun -n 1 megatop-compsep-plot --config ${PARAM_FILE}


echo ""
echo ""
echo "Plotting spectra estimater outputs"
mpirun -n 1 megatop-map2cl-plot --config ${PARAM_FILE}


echo ""
echo ""
echo "Plotting noise spectra estimater outputs"
mpirun -n 1 megatop-noisespectra-plot --config ${PARAM_FILE}


echo ""
echo ""
echo "Plotting r statistics"
mpirun -n 1 megatop-cl2r-plot --config ${PARAM_FILE}
echo ""
echo "Plotting mcmc results statistics"
mpirun -n 1 megatop-cl2r_mcmc-plot --config ${PARAM_FILE}