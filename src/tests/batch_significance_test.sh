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

echo "Doing statistical Tests on G and F with G test set"
./significance_test.py "${OUTPUT_DIR}/model_903/set_G/predictions/903_G_fold_metrics.csv" "${OUTPUT_DIR}/model_903/set_F/cross_eval/set_G/903_F_with_G_fold_metrics.csv"
echo
echo "Doing statistical Tests on G and M with G test set"
./significance_test.py "${OUTPUT_DIR}/model_903/set_G/predictions/903_G_fold_metrics.csv" "${OUTPUT_DIR}/model_903/set_M/cross_eval/set_G/903_M_with_G_fold_metrics.csv"
echo
echo "Doing statistical Tests on F and M with F test set"
./significance_test.py "${OUTPUT_DIR}/model_903/set_F/predictions/903_F_fold_metrics.csv" "${OUTPUT_DIR}/model_903/set_M/cross_eval/set_F/903_M_with_F_fold_metrics.csv"
echo
echo "Doing statistical Tests on M and F with M test set"
./significance_test.py "${OUTPUT_DIR}/model_903/set_M/predictions/903_M_fold_metrics.csv" "${OUTPUT_DIR}/model_903/set_F/cross_eval/set_M/903_F_with_M_fold_metrics.csv"
echo
echo "Doing statistical Tests on F with F and F with M test set"
./significance_test.py "${OUTPUT_DIR}/model_903/set_F/predictions/903_F_fold_metrics.csv" "${OUTPUT_DIR}/model_903/set_F/cross_eval/set_M/903_F_with_M_fold_metrics.csv"
echo
echo "Doing statistical Tests on M with M and M with F test set"
./significance_test.py "${OUTPUT_DIR}/model_903/set_M/predictions/903_M_fold_metrics.csv" "${OUTPUT_DIR}/model_903/set_M/cross_eval/set_F/903_M_with_F_fold_metrics.csv"
echo
