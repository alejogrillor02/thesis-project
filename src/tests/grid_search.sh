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

./grid_search.py "903" "G" "1"
