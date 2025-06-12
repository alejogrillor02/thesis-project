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
export PROJECT_ROOT="$PROJECTDIR"

model_strings=($(yq -r '.MODELS[]' ${PROJECTDIR}/config.yaml))
set_strings=($(yq -r '.SETS[]' ${PROJECTDIR}/config.yaml))
folds=$(yq -r '.N_FOLDS' ${PROJECTDIR}/config.yaml)

for str1 in "${model_strings[@]}"; do
	for str2 in "${set_strings[@]}"; do
		for ((i = 1; i <= folds; i++)); do
			./train.py ${str1} ${str2} $i && echo "Done training $str1 model for $str2 set, fold $i."
			echo -n '\n'
		done
	done
done
