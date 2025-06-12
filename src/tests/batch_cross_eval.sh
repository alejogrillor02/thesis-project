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

for str1 in "${model_strings[@]}"; do
    for str2 in "${set_strings[@]}"; do
        for str3 in "${set_strings[@]}"; do
            if [ $str2 == $str3 ]; then
                true # Do nothing
            else
                ${PROJECTDIR}/src/eval/evaluate.py "${str1}" "${str2}" "${str3}" && echo "Done evaluating $str1 model for $str2 set with $str3 test set."
                echo
            fi
        done
    done
done
