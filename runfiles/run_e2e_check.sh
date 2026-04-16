#!/bin/bash
# End-to-end sanity check for the Megatop pipeline.
# Runs all computation steps in order without plots.
# Intended for quick validation with a small n_sim config.
#
# Usage:
#   ./runfiles/run_e2e_check.sh <config.yaml>

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.yaml>" >&2
    exit 1
fi

PARAM_FILE="$1"

if [[ ! -f "$PARAM_FILE" ]]; then
    echo "Config file not found: $PARAM_FILE" >&2
    exit 1
fi

echo "Running e2e check with config: ${PARAM_FILE}"
echo ""

step() {
    echo "--- $* ---"
}

step "mask-handler"
megatop-mask-run --config "${PARAM_FILE}"

step "binning-maker"
megatop-binning-run --config "${PARAM_FILE}"

step "mocker"
megatop-mock-run --config "${PARAM_FILE}"

step "preprocesser"
megatop-preproc-run --config "${PARAM_FILE}"

step "noise-covariance"
megatop-noisecov-run --config "${PARAM_FILE}"

step "component-separation"
megatop-compsep-run --config "${PARAM_FILE}"

step "map-to-cl"
megatop-map2cl-run --config "${PARAM_FILE}"

step "noise-spectra"
megatop-noisespectra-run --config "${PARAM_FILE}"

step "cl-to-r"
megatop-cl2r-run --config "${PARAM_FILE}"

echo ""
echo "e2e check complete."
