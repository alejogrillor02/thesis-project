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
export PROJECT_ROOT="$PROJECTDIR"

DATA_DIR=${PROJECTDIR}/$(yq -r '.DATA_DIR' ${PROJECTDIR}/config.yaml)

model_strings=($(yq -r '.MODELS[]' ${PROJECTDIR}/config.yaml))
set_strings=($(yq -r '.SETS[]' ${PROJECTDIR}/config.yaml))

for str1 in "${model_strings[@]}"; do
	./filter_csv.py "${DATA_DIR}/raw/${str1}.csv" && echo "Done filtering $str1.csv"
	./process_csv.py "${DATA_DIR}/processed/filtered/${str1}_filtered.csv" && echo "Done processing $str1 model."
	# mv "${DATADIR}/processed/${str1}_norm_stats.csv" ""
	for str2 in "${set_strings[@]}"; do
		./kfold.py "${DATA_DIR}/processed/${str1}_${str2}.txt" && echo "Done sectioning $str1 model for $str2 set."
		mv "${DATA_DIR}/processed/${str1}_${str2}_test.txt" "${DATA_DIR}/train/model_${str1}/set_${str2}"
	done
done
