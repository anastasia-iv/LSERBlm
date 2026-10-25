#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON:-python}"
new_tokens="${1:-data/serbian_expanded_v3/bpe8k}"
gpu_a="${2:-0}"
gpu_b="${3:-1}"
parent="runs/serbian_283m_2/best.pt"
config="configs/continued_deep.json"
old_tokens="data/serbian_mix_v2/bpe8k"

if [[ "$gpu_a" == "$gpu_b" ]]; then
  echo 'Choose different GPUs for the two concurrent branches.' >&2
  exit 1
fi
for path in "$parent" "$config" "$old_tokens/tokenizer.json" "$new_tokens/tokenizer.json" \
            "$old_tokens/train.bin" "$new_tokens/train.bin" \
            "$old_tokens/validation.bin" "$new_tokens/validation.bin"; do
  [[ -f "$path" ]] || { echo "Missing file: $path" >&2; exit 1; }
done
mkdir -p runs/logs
for branch in A B; do
  run="runs/deep_continue_${branch}"
  log="runs/logs/deep_continue_${branch}.log"
  if [[ -e "$log" || ( -d "$run" && -n "$(ls -A -- "$run")" ) ]]; then
    echo "Existing run/log for branch $branch. Resume it explicitly; this script starts new runs." >&2
    exit 1
  fi
done
"$python_bin" - "$old_tokens" "$new_tokens" <<'PY'
import sys
from pathlib import Path
from serbian_llm.utils import file_sha256, read_json
a, b = map(Path, sys.argv[1:])
if file_sha256(a / 'tokenizer.json') != file_sha256(b / 'tokenizer.json'):
    raise SystemExit('Branches must use exactly the same tokenizer.')
old, new = (read_json(p / 'token_stats.json') for p in [a, b])
for split in ['validation', 'test']:
    if old['corpus_files'][split] != new['corpus_files'][split]:
        raise SystemExit('Branches must use the same frozen validation/test.')
if old['corpus_files']['train'] == new['corpus_files']['train']:
    raise SystemExit('Branch B has the same train as A; encode the expanded corpus first.')
PY
for branch in A B; do
  if [[ "$branch" == A ]]; then tokens="$old_tokens"; gpu="$gpu_a";
  else tokens="$new_tokens"; gpu="$gpu_b"; fi
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$python_bin" -u -m serbian_llm continue-train \
    --config "$config" --init-from "$parent" --tokenizer "$tokens" \
    --output "runs/deep_continue_${branch}" --device cuda:0 \
    > "runs/logs/deep_continue_${branch}.log" 2>&1 &
  pid=$!
  echo "$pid" > "runs/logs/deep_continue_${branch}.pid"
  echo "Branch $branch: PID $pid, GPU $gpu, runs/logs/deep_continue_${branch}.log"
done
