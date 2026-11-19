#!/usr/bin/env bash

set -uo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: PlotPlanBatch.sh [-R] [-s|--save SAVE_PATTERN] FOLDER [PLOT_OPTIONS...]

SAVE_PATTERN is a folder or a folder/pattern containing '%' as a placeholder
for the input basename (without its extension). The '.png' extension is added
when the pattern has no extension.
EOF
}

recursive=false
save_spec=""
folder=""
plot_options=()

while (($#)); do
    case "$1" in
        -R)
            recursive=true
            shift
            ;;
        -s|--save)
            if (($# < 2)); then
                echo "Missing value for $1" >&2
                usage
                exit 2
            fi
            save_spec=$2
            shift 2
            ;;
        -s=*|--save=*)
            save_spec=${1#*=}
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ -z "$folder" && -d "$1" ]]; then
                folder=$1
            else
                plot_options+=("$1")
            fi
            shift
            ;;
    esac
done

if [[ -z "$folder" ]]; then
    echo "A source folder is required." >&2
    usage
    exit 2
fi
if [[ ! -d "$folder" ]]; then
    echo "Source folder does not exist: $folder" >&2
    exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_script="$script_dir/PlotPlan.py"

tmp_root=${TMPDIR:-/tmp}
tmp_folder=$(mktemp -d "$tmp_root/dfp_ui_batch.XXXXXX")
trap 'rm -rf "$tmp_folder"' EXIT

if [[ -n "$save_spec" ]]; then
    if [[ "$save_spec" != *%* ]]; then
        echo "Save argument must contain a '%' output pattern: $save_spec" >&2
        exit 2
    fi
    save_spec=$(mkdir -p "$(dirname "$save_spec")" && printf '%s' "$save_spec")
else
    save_spec="$tmp_folder/%"
fi

csv_escape() {
    local value=$1
    value=${value//\"/\"\"}
    printf '"%s"' "$value"
}

printf 'source_file,output_file,aspect_ratio,compatible\n'

declare -a source_files=()
if $recursive; then
    while IFS= read -r -d '' file; do
        source_files+=("$file")
    done < <(find "$folder" -type f \( -iname '*.csv' -o -iname '*.json' \) -print0)
else
    while IFS= read -r -d '' file; do
        source_files+=("$file")
    done < <(find "$folder" -mindepth 1 -maxdepth 1 -type f \( -iname '*.csv' -o -iname '*.json' \) -print0)
fi
IFS=$'\n' source_files=($(printf '%s\n' "${source_files[@]}" | sort))
unset IFS

requested_ratio=""
for ((i = 0; i < ${#plot_options[@]}; i++)); do
    if [[ "${plot_options[$i]}" == "--ratio" && $((i + 1)) -lt ${#plot_options[@]} ]]; then
        requested_ratio=${plot_options[$((i + 1))]}
    elif [[ "${plot_options[$i]}" == --ratio=* ]]; then
        requested_ratio=${plot_options[$i]#*=}
    fi
done

for source_file in "${source_files[@]}"; do
    source_name=$(basename "$source_file")
    source_stem=${source_name%.*}

    output_path=${save_spec//%/$source_stem}
    if [[ "$output_path" != *.png && "$output_path" != *.gif && "$output_path" != *.webp && "$output_path" != *.mp4 && "$output_path" != *.webm && "$output_path" != *.apng ]]; then
        output_path+=".png"
    fi
    mkdir -p "$(dirname "$output_path")"

    run_output="$tmp_folder/run_output"
    : > "$run_output"

    PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}" \
        python3 "$python_script" "$source_file" "$output_path" --no-print \
        "${plot_options[@]}" > >(tee "$run_output" >&2) 2>&1
    status=$?

    computed_ratio=$requested_ratio
    if grep -q 'Computed aspect ratio:' "$run_output"; then
        computed_ratio=$(sed -n 's/^Computed aspect ratio: //p' "$run_output" | tail -n 1)
    fi

    compatible=1
    if ((status == 2)) || [[ $status -ne 0 && -n "$requested_ratio" ]]; then
        compatible=0
    fi

    printf '%s,%s,%s,%d\n' \
        "$(csv_escape "$source_file")" \
        "$(csv_escape "$output_path")" \
        "$(csv_escape "$computed_ratio")" \
        "$compatible"
done
