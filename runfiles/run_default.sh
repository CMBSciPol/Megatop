#!/bin/bash
export FI_PROVIDER=tcp
PARAM_FILE="../paramfiles/default_config.yaml"

#echo 'export SO_NOMINAL_HITMAP_PATH=/tmp/so_nominal_hitmap.fits' >> ~/.zshrc
echo "Running pipeline with paramfile: ${PARAM_FILE}"

conda init bash
source ~/.bashrc
conda activate megatop


echo "------------------------------------------------------------"
echo "|                        MASK-HANDLER                      |"
echo "------------------------------------------------------------"
megatop-mask-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting mask outputs"
megatop-mask-plot --config ${PARAM_FILE}

echo "------------------------------------------------------------"
echo "|                       BINNING-MAKER                      |"
echo "------------------------------------------------------------"
megatop-binning-run --config ${PARAM_FILE}
echo ""
echo ""

echo "------------------------------------------------------------"
echo "|                           MOCKER                         |"
echo "------------------------------------------------------------"
mpirun -n 4 python $(which megatop-mock-run) --config ${PARAM_FILE}
#megatop-mock-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting mocker outputs"
megatop-mock-plot --config ${PARAM_FILE}

echo "------------------------------------------------------------"
echo "|            TRANSFER FUNCTION COMPUTATION                  |"
echo "------------------------------------------------------------"
#megatop-TFcomputing-run --config ${PARAM_FILE}
echo ""
echo ""

echo "------------------------------------------------------------"
echo "|                       PRE-PROCESSER                      |"
echo "------------------------------------------------------------"
mpirun -n 4 python $(which megatop-preproc-run) --config ${PARAM_FILE}
#megatop-preproc-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting pre-processer outputs"
megatop-preproc-plot --config ${PARAM_FILE}

echo "------------------------------------------------------------"
echo "|                  NOISE PREPROCESSING                     |"
echo "------------------------------------------------------------"
mpirun -n 4 python $(which megatop-noise-preproc-run) --config ${PARAM_FILE}
#megatop-noise-preproc-run --config ${PARAM_FILE}

echo "------------------------------------------------------------"
echo "|                NOISE-COVARIANCE COMPUTATION              |"
echo "------------------------------------------------------------"
mpirun -n 2 python $(which megatop-noisecov-run) --config ${PARAM_FILE}megatop-noisecov-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting noise covariance outputs"
megatop-noisecov-plot --config ${PARAM_FILE}

echo "------------------------------------------------------------"
echo "|                    COMPONENT SEPARATION                  |"
echo "------------------------------------------------------------"
mpirun -n 4 python $(which megatop-compsep-run) --config ${PARAM_FILE}
#megatop-compsep-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting component separater outputs"
#mpirun -n 8 python $(which megatop-compsep-plot) --config ${PARAM_FILE}
megatop-compsep-plot --config ${PARAM_FILE}

echo "------------------------------------------------------------"
echo "|                     SPECTRA ESTIMATION                   |"
echo "------------------------------------------------------------"
mpirun -n 2 python $(which megatop-map2cl-run) --config ${PARAM_FILE}
#megatop-map2cl-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting spectra estimater outputs"
megatop-map2cl-plot --config ${PARAM_FILE}

echo "------------------------------------------------------------"
echo "|                  NOISE SPECTRA ESTIMATION                |"
echo "------------------------------------------------------------"
mpirun -n 2 python $(which megatop-noisespectra-run) --config ${PARAM_FILE}
#for i in 47 48 49; do
#    mpirun -n 4 python $(which megatop-noisespectra-run) --config ${PARAM_FILE} --sim $i
#done
#megatop-noisespectra-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting noise spectra estimater outputs"
megatop-noisespectra-plot --config ${PARAM_FILE}

echo "------------------------------------------------------------"
echo "|            COSMOLOGICAL PARAMETERS ESTIMATION            |"
echo "------------------------------------------------------------"
mpirun -n 4 python $(which megatop-cl2r-run) --config ${PARAM_FILE}
#for i in 47 48 49; do
#    mpirun -n 2 python $(which megatop-cl2r-run) --config ${PARAM_FILE} --sim $i
#done
#megatop-cl2r-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting r statistics"
megatop-cl2r-plot --config ${PARAM_FILE}
echo ""
echo "Plotting mcmc results statistics"
megatop-cl2r_mcmc-plot --config ${PARAM_FILE}
