#!/bin/bash

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
cd "$PROJECTDIR"

repetitions=5

for ((i = 1; i <= $repetitions; i++)); do
    ./src/data/batch_preprocessing.sh
    ./src/train/batch_training.sh
    ./src/eval/batch_eval.sh
    echo
    echo "FINISHED ${i}-th Repetition"
    echo
done

OUTPUT_DIR="${PROJECTDIR}/$(yq -r '.OUTPUT_DIR' ${PROJECTDIR}/config.yaml)"

Model1="${OUTPUT_DIR}/model_903/set_E/predictions/903_E_metrics.csv"
Model2="${OUTPUT_DIR}/model_903/set_G/predictions/903_G_metrics.csv"
export PROJECT_ROOT="$PROJECTDIR"

./src/tests/significance_test.py "${Model1}" "${Model2}"
