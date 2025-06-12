#!/bin/bash
# Author: alejo.grillor02

SCRDIR=$(realpath "$(dirname "$0")")
cd "$SCRDIR" || exit

PROJECTDIR="$SCRDIR"
while [[ ! -f "${PROJECTDIR}/config.yaml" ]] && [[ "$PROJECTDIR" != "/" ]]; do
    PROJECTDIR=$(dirname "$PROJECTDIR")
done

if [[ ! -f "${PROJECTDIR}/config.yaml" ]]; then
    echo "Error: config.yaml not found in project root!" >&2
    exit 1
fi
export PROJECT_ROOT="$PROJECTDIR"

OUTPUT_DIR=${PROJECTDIR}/$(yq -r '.OUTPUT_DIR' ${PROJECTDIR}/config.yaml)
MODEL_DIR=${PROJECTDIR}/$(yq -r '.MODEL_DIR' ${PROJECTDIR}/config.yaml)
DATA_DIR=${PROJECTDIR}/$(yq -r '.DATA_DIR' ${PROJECTDIR}/config.yaml)
TRAINDATA_DIR="${DATA_DIR}/train"

./significance_test.py "/home/hope/Documents/Thesis/project/output/model_903/set_E/predictions/903_E_fold_errors.csv" "/home/hope/Documents/Thesis/project/output/model_903/set_G/cross_eval/set_E/903_G_with_E_fold_errors.csv"
