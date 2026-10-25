#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

model="${1:?Usage: train_model.sh MODEL [GPU] [OUTPUT]}"
gpu="${2:-2}"
output="${3:-runs/${model}_new}"
python_bin="${PYTHON:-python}"
command=train
tokenizer=data/serbian_mix_v2/bpe8k
parent_args=()
case "$model" in
  absolute_41m) config=configs/ablation_absolute.json ;;
  rope_41m) config=configs/main_rope.json ;;
  serbian_283m_1) config=configs/main_rope_1.json ;;
  serbian_283m_2) config=configs/main_rope_2.json ;;
  serbian_283m_3) config=configs/main_rope_3.json ;;
  deep_continue_A|deep_continue_B)
    config=configs/continued_deep.json
    command=continue-train
    parent_args=(--init-from runs/serbian_283m_2/best.pt)
    if [[ "$model" == deep_continue_B ]]; then tokenizer=data/serbian_expanded_v3/bpe8k; fi
    ;;
  *) echo "Unknown run: $model" >&2; exit 1 ;;
esac
log="runs/logs/$(basename -- "$output").log"
mkdir -p -- runs/logs "$(dirname -- "$output")"
if [[ -e "$log" || ( -d "$output" && -n "$(ls -A -- "$output")" ) ]]; then
  echo "Output/log exists. Choose a new output or resume explicitly." >&2
  exit 1
fi
for path in "$config" "$tokenizer/tokenizer.json" "$tokenizer/train.bin" "$tokenizer/validation.bin"; do
  [[ -f "$path" ]] || { echo "Missing file: $path" >&2; exit 1; }
done
CUDA_VISIBLE_DEVICES="$gpu" nohup "$python_bin" -u -m serbian_llm "$command" \
  --config "$config" --tokenizer "$tokenizer" --output "$output" \
  "${parent_args[@]}" --device cuda:0 > "$log" 2>&1 &
echo "PID $!, GPU $gpu, log $log"
