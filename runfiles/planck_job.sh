
#!/bin/bash
#PBS -N planck_megatop
#PBS -o /home/cgubeno/log/
#PBS -e /home/cgubeno/log/
#PBS -l select=1:ncpus=52:mem=375gb
#PBS -l walltime=12:00:00
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

PARAM_FILE="/home/cgubeno/Megatop/paramfiles/planck_wn_600.yaml"

echo "Running pipeline with paramfile: ${PARAM_FILE}"


echo "------------------------------------------------------------"
echo "|                        MASK-HANDLER                      |"
echo "------------------------------------------------------------"
mpirun -n 1 megatop-mask-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting mask outputs"
mpirun -n 1 megatop-mask-plot --config ${PARAM_FILE}

echo "------------------------------------------------------------"
echo "|                       BINNING-MAKER                      |"
echo "------------------------------------------------------------"
mpirun -n 1 megatop-binning-run --config ${PARAM_FILE}
echo ""
echo ""

echo "------------------------------------------------------------"
echo "|                           MOCKER                         |"
echo "------------------------------------------------------------"
mpirun -n 50 megatop-mock-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting mocker outputs"
mpirun -n 1 megatop-mock-plot --config ${PARAM_FILE}

echo "------------------------------------------------------------"
echo "|            TRANSFER FUNCTION COMPUTATION                  |"
echo "------------------------------------------------------------"
mpirun -n 1 megatop-TFcomputing-run --config ${PARAM_FILE}
echo ""
echo ""

echo "------------------------------------------------------------"
echo "|                       PRE-PROCESSER                      |"
echo "------------------------------------------------------------"
mpirun -n 50 megatop-preproc-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting pre-processer outputs"
mpirun -n 1 megatop-preproc-plot --config ${PARAM_FILE}

echo "------------------------------------------------------------"
echo "|                NOISE-COVARIANCE COMPUTATION              |"
echo "------------------------------------------------------------"
mpirun -n 50 megatop-noisecov-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting noise covariance outputs"
mpirun -n 1 megatop-noisecov-plot --config ${PARAM_FILE}

echo "------------------------------------------------------------"
echo "|                    COMPONENT SEPARATION                  |"
echo "------------------------------------------------------------"
mpirun -n 50 megatop-compsep-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting component separater outputs"
mpirun -n 1 megatop-compsep-plot --config ${PARAM_FILE}

echo "------------------------------------------------------------"
echo "|                     SPECTRA ESTIMATION                   |"
echo "------------------------------------------------------------"
mpirun -n 50 megatop-map2cl-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting spectra estimater outputs"
mpirun -n 1 megatop-map2cl-plot --config ${PARAM_FILE}

echo "------------------------------------------------------------"
echo "|                  NOISE SPECTRA ESTIMATION                |"
echo "------------------------------------------------------------"
mpirun -n 50 megatop-noisespectra-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting noise spectra estimater outputs"
mpirun -n 1 megatop-noisespectra-plot --config ${PARAM_FILE}

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