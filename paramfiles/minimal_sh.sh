#!/bin/bash
PARAM_FILE="./paramfiles/minimal_test_ben_sims.yml"

echo $PARAM_FILE

#mask handling
# megatop-mask-run --globals $PARAM_FILE
# megatop-mask-plot --globals $PARAM_FILE

# #mocking data
# megatop-mock-run --globals $PARAM_FILE

#pre-processing signal+noise simu
# megatop-preproc-run --globals $PARAM_FILE
# megatop-preproc-plot --globals $PARAM_FILE

# estimate noise_cov
# mpirun -n 8 megatop-noisecov-run --globals $PARAM_FILE

# # #compsep
megatop-compsep --globals $PARAM_FILE
