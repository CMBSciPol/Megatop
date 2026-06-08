#!/bin/bash
#PBS -N heavy_npipe
#PBS -o /home/cgubeno/log/
#PBS -e /home/cgubeno/log/
#PBS -l select=2:ncpus=52:mem=375gb
#PBS -l walltime=700:00:00
#PBS -u cgubeno
#PBS -m ae
#PBS -q small
#PBS -k doe

source ~/.bashrc
conda activate megatop

PARAM_FILE="/home/cgubeno/Megatop/paramfiles/various_test.yaml"

echo "Running pipeline with paramfile: ${PARAM_FILE}"
echo ""

# echo "------------------------------------------------------------"
# echo "|                        MASK-HANDLER                      |"
# echo "------------------------------------------------------------"
# mpirun -np 1 megatop-mask-run --config ${PARAM_FILE}
# echo ""
# echo ""
# echo "Plotting mask outputs"
# mpirun -np 1 megatop-mask-plot --config ${PARAM_FILE}

# echo "------------------------------------------------------------"
# echo "|                       BINNING-MAKER                      |"
# echo "------------------------------------------------------------"
# mpirun -np 1 megatop-binning-run --config ${PARAM_FILE}
# echo ""
# echo ""

# echo "------------------------------------------------------------"
# echo "|                           MOCKER                         |"
# echo "------------------------------------------------------------"
# mpirun -np 50 megatop-mock-run --config ${PARAM_FILE}
# echo ""
# echo ""
# echo "Plotting mocker outputs"
# mpirun -np 1 megatop-mock-plot --config ${PARAM_FILE}

# echo "------------------------------------------------------------"
# echo "|            TRANSFER FUNCTION COMPUTATION                 |"
# echo "------------------------------------------------------------"
# mpirun -np 1 megatop-TFcomputing-run --config ${PARAM_FILE}
# echo ""
# echo ""

echo "------------------------------------------------------------"
echo "|                       PRE-PROCESSER                      |"
echo "------------------------------------------------------------"
mpirun -np 100 megatop-preproc-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting pre-processer outputs"
mpirun -np 1 megatop-preproc-plot --config ${PARAM_FILE}

echo "------------------------------------------------------------"
echo "|                  NOISE PREPROCESSING                     |"
echo "------------------------------------------------------------"
mpirun -np 100 megatop-noise-preproc-run --config ${PARAM_FILE}
echo ""
echo ""

# echo "------------------------------------------------------------"
# echo "|                NOISE-COVARIANCE COMPUTATION              |"
# echo "------------------------------------------------------------"
# mpirun -np 50 megatop-noisecov-run --config ${PARAM_FILE}
# echo ""
# echo ""
# echo "Plotting noise covariance outputs"
# mpirun -np 1 megatop-noisecov-plot --config ${PARAM_FILE}

# echo "------------------------------------------------------------"
# echo "|                    COMPONENT SEPARATION                  |"
# echo "------------------------------------------------------------"
# mpirun -np 50 megatop-compsep-run --config ${PARAM_FILE}
# echo ""
# echo ""
# echo "Plotting component separater outputs"
# mpirun -np 1 megatop-compsep-plot --config ${PARAM_FILE}

# echo "------------------------------------------------------------"
# echo "|                     SPECTRA ESTIMATION                   |"
# echo "------------------------------------------------------------"
# mpirun -np 50 megatop-map2cl-run --config ${PARAM_FILE}
# echo ""
# echo ""
# echo "Plotting spectra estimater outputs"
# mpirun -np 1 megatop-map2cl-plot --config ${PARAM_FILE}

# echo "------------------------------------------------------------"
# echo "|                  NOISE SPECTRA ESTIMATION                |"
# echo "------------------------------------------------------------"
# mpirun -n 16 megatop-noisespectra-run --config ${PARAM_FILE}
# echo ""
# echo ""
# echo "Plotting noise spectra estimater outputs"
# mpirun -np 1 megatop-noisespectra-plot --config ${PARAM_FILE}

# echo "------------------------------------------------------------"
# echo "|            COSMOLOGICAL PARAMETERS ESTIMATION            |"
# echo "------------------------------------------------------------"
# mpirun -np 50 megatop-cl2r-run --config ${PARAM_FILE}
# echo ""
# echo ""
# echo "Plotting r statistics"
# mpirun -np 1 megatop-cl2r-plot --config ${PARAM_FILE}
# echo ""
# echo "Plotting mcmc results statistics"
# mpirun -n 1 megatop-cl2r_mcmc-plot --config ${PARAM_FILE}