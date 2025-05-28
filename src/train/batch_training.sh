#!/bin/bash
# Author: alejo.grillor02

## Esto es para que el output se redirija a STDOUT y a un archivo a la vez
# exec {fd}>&1
# exec > >(tee batch.log)

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

DATA_DIR=${PROJECTDIR}/$(yq -r '.DATA_DIR' ${PROJECTDIR}/config.yaml)
TRAINDATA_DIR="${DATA_DIR}/train"
MODEL_DIR=${PROJECTDIR}/$(yq -r '.MODEL_DIR' ${PROJECTDIR}/config.yaml)

model_strings=($(yq -r '.MODELS[]' ${PROJECTDIR}/config.yaml))
set_strings=($(yq -r '.SETS[]' ${PROJECTDIR}/config.yaml))
folds=$(yq -r '.FOLDS' ${PROJECTDIR}/config.yaml)
epochs=$(yq -r '.EPOCHS' ${PROJECTDIR}/config.yaml)
batch_size=$(yq -r '.BATCH_SIZE' ${PROJECTDIR}/config.yaml)


for str1 in "${model_strings[@]}"; do
	for str2 in "${set_strings[@]}"; do
		for ((i=1; i<=folds; i++)); do
			./train.py "${TRAINDATA_DIR}/model_${str1}/set_${str2}" $i $folds $epochs $batch_size "${MODEL_DIR}" && echo "Done training $str model for $str2 set, $i fold."
		done
	done
done
