#!/bin/bash


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
python get_noise_cov_pixel.py --globals ${paramfile} --verbose --plots


echo "------------------------------------------------------------"
echo "|                    COMPONENT SEPARATION                  |"
echo "------------------------------------------------------------"
python parametric_separation.py --globals ${paramfile} --verbose --plots


echo "------------------------------------------------------------"
echo "|                     SPECTRA ESTIMATION                   |"
echo "------------------------------------------------------------"
python map_to_cl.py --globals ${paramfile} --verbose --plots