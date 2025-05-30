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

OUTPUT_DIR=${PROJECTDIR}/$(yq -r '.OUTPUT_DIR' ${PROJECTDIR}/config.yaml)
MODEL_DIR=${PROJECTDIR}/$(yq -r '.MODEL_DIR' ${PROJECTDIR}/config.yaml)
DATA_DIR=${PROJECTDIR}/$(yq -r '.DATA_DIR' ${PROJECTDIR}/config.yaml)
TRAINDATA_DIR="${DATA_DIR}/train"
NORM_DATA_DIR="${DATA_DIR}/processed"

model_strings=($(yq -r '.MODELS[]' ${PROJECTDIR}/config.yaml))
set_strings=($(yq -r '.SETS[]' ${PROJECTDIR}/config.yaml))
folds=$(yq -r '.FOLDS' ${PROJECTDIR}/config.yaml)


for str1 in "${model_strings[@]}"; do
	for str2 in "${set_strings[@]}"; do
		./evaluate.py "${MODEL_DIR}/model_${str1}/set_${str2}" "${TRAINDATA_DIR}/model_${str1}/set_${str2}/${str1}_${str2}_test.txt" $folds "${NORM_DATA_DIR}" "${OUTPUT_DIR}" && echo "Done evaluating $str1 model for $str2 set."
	done
done