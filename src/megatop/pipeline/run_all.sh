paramfile='../paramfiles/config_paper_test.yml'

echo "Running pipeline with paramfile: ${paramfile}"

echo "------------------------------------------------------------"
echo "|           PREPARING METADATA AND SIMULATIONS             |"
echo "------------------------------------------------------------"

echo "Pre-processing data..."
echo "-------------------"

python pre_processing.py --globals ${paramfile}
python get_noise_cov.py --globals ${paramfile}
