#!/bin/bash
PARAM_FILE="../paramfiles/default_config.yaml"

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
echo "|                           MOCKER                         |"
echo "------------------------------------------------------------"
megatop-mock-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting mocker outputs"
megatop-mock-plot --config ${PARAM_FILE}


echo "------------------------------------------------------------"
echo "|                       PRE-PROCESSER                      |"
echo "------------------------------------------------------------"
megatop-preproc-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting pre-processer outputs"
megatop-preproc-plot --config ${PARAM_FILE}


echo "------------------------------------------------------------"
echo "|                NOISE-COVARIANCE COMPUTATION              |"
echo "------------------------------------------------------------"
megatop-noisecov-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting noise covariance outputs"
megatop-noisecov-plot --config ${PARAM_FILE}


echo "------------------------------------------------------------"
echo "|                    COMPONENT SEPARATION                  |"
echo "------------------------------------------------------------"
megatop-compsep-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting component separater outputs"
megatop-compsep-plot --config ${PARAM_FILE}


echo "------------------------------------------------------------"
echo "|                     SPECTRA ESTIMATION                   |"
echo "------------------------------------------------------------"
megatop-map2cl-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting spectra estimater outputs"
megatop-map2cl-plot --config ${PARAM_FILE}


echo "------------------------------------------------------------"
echo "|                  NOISE SPECTRA ESTIMATION                |"
echo "------------------------------------------------------------"
megatop-noisespectra-run --config ${PARAM_FILE}
echo ""
echo ""
echo "Plotting noise spectra estimater outputs"
megatop-noisespectra-plot --config ${PARAM_FILE}


echo "------------------------------------------------------------"
echo "|            COSMOLOGICAL PARAMETERS ESTIMATION            |"
echo "------------------------------------------------------------"
echo "TO BE IMPLEMENTED"
