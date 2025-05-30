#!/bin/bash
# author: alejo.grillor02

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

DATADIR=${PROJECTDIR}/$(yq -r '.DATA_DIR' ${PROJECTDIR}/config.yaml)
model_strings=($(yq -r '.MODELS[]' ${PROJECTDIR}/config.yaml))
set_strings=($(yq -r '.SETS[]' ${PROJECTDIR}/config.yaml))
folds=$(yq -r '.FOLDS' ${PROJECTDIR}/config.yaml)
features=($(yq -r '.FEATURES[]' ${PROJECTDIR}/config.yaml))

for str1 in "${model_strings[@]}"; do
		./filter_csv.py "${DATADIR}/raw/${str1}.csv" "${DATADIR}/processed/filtered" && echo "Done filtering $str1.csv"
		./process_csv.py "${DATADIR}/processed/filtered/${str1}_filtered.csv" "${DATADIR}/processed" "${features[@]}" && echo "Done processing $str1 model."
		for str2 in "${set_strings[@]}"; do
				./kfold.py "${DATADIR}/processed/${str1}_${str2}.txt" $folds "${DATADIR}/train" "${features[@]}" && echo "Done sectioning $str1 model for $str2 set."
				mv "${DATADIR}/processed/${str1}_${str2}_test.txt" "${DATADIR}/train/model_${str1}/set_${str2}"
		done
done