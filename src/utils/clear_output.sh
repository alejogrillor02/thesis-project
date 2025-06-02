#!/bin/bash
#
# About: Simple bash script for saving output
# Author: alejo.grillor02
# License: GNU GPLv3

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
MODEL_DIR=${PROJECTDIR}/$(yq -r '.MODEL_DIR' ${PROJECTDIR}/config.yaml)
OUTPUT_DIR=${PROJECTDIR}/$(yq -r '.OUTPUT_DIR' ${PROJECTDIR}/config.yaml)

bck_path="${PROJECTDIR}/.old/$(date '+%F %H.%M.%S')"
paths=(
	"${DATA_DIR}/train"
	"${DATA_DIR}/processed"
	"${MODEL_DIR}"
	"${OUTPUT_DIR}"
)

mkdir -pv "$bck_path"

for dir in "${paths[@]}"; do
	mv -v "$dir" "$bck_path" || echo "The output dir $dir doesn't exist"
done
