#!/bin/bash
# End-to-end sanity check for the Megatop pipeline.
# Runs all computation steps in order without plots.
# Intended for quick validation with a small n_sim config.
#
# Usage:
#   ./runfiles/run_e2e_check.sh <config.yaml> [--np N]
#
# Options:
#   --np N   Run each step under "mpirun -n N" to exercise MPI code paths.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.yaml> [--np N]" >&2
    exit 1
fi

PARAM_FILE="$1"
shift

NP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --np)
            NP="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "$PARAM_FILE" ]]; then
    echo "Config file not found: $PARAM_FILE" >&2
    exit 1
fi

if [[ -n "$NP" ]]; then
    RUN="mpirun -n ${NP} --oversubscribe"
    echo "Running e2e check with config: ${PARAM_FILE} (MPI, ${NP} processes)"
else
    RUN=""
    echo "Running e2e check with config: ${PARAM_FILE}"
fi
echo ""

step() {
    echo "--- $* ---"
}

step "mask-handler"
megatop-mask-run --config "${PARAM_FILE}"

step "binning-maker"
megatop-binning-run --config "${PARAM_FILE}"

step "mocker"
${RUN} megatop-mock-run --config "${PARAM_FILE}"

step "preprocesser"
${RUN} megatop-preproc-run --config "${PARAM_FILE}"

step "noise-covariance"
${RUN} megatop-noisecov-run --config "${PARAM_FILE}"

step "component-separation"
${RUN} megatop-compsep-run --config "${PARAM_FILE}"

step "map-to-cl"
${RUN} megatop-map2cl-run --config "${PARAM_FILE}"

step "noise-spectra"
${RUN} megatop-noisespectra-run --config "${PARAM_FILE}"

step "cl-to-r"
${RUN} megatop-cl2r-run --config "${PARAM_FILE}"

echo ""
echo "e2e check complete."
