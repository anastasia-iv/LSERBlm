#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON:-python}"
variant="${1:?Usage: run_sft.sh lr1e5|lr3e5}"
case "$variant" in lr1e5|lr3e5) ;; *) echo 'Unknown SFT variant' >&2; exit 1 ;; esac
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
data=data/sft_euroalpaca_v2
output="runs/deep_continue_B_sft_${variant}"

"$python_bin" -u -m serbian_llm sft \
  --config "configs/sft_deep_${variant}.json" \
  --tokenizer data/serbian_mix_v2/bpe8k \
  --init-from runs/deep_continue_B/best.pt \
  --train-jsonl "$data/train.jsonl" --validation-jsonl "$data/validation.jsonl" \
  --output "$output" --device cuda:0

for split in validation test; do
  "$python_bin" -u -m serbian_llm evaluate-sft \
    --checkpoint "$output/best.pt" --data "$data/$split.jsonl" \
    --output "$output/sft_$split.json" --device cuda:0 --precision bf16
done

"$python_bin" -u -m serbian_llm evaluate \
  --checkpoint "$output/best.pt" --data data/serbian_mix_v2/corpus/validation.jsonl \
  --output "$output/lm_validation.json" --limit 0 --stride 256 --device cuda:0 --precision bf16

for protocol in raw sft; do
  "$python_bin" -u -m serbian_llm benchmark \
    --checkpoint "$output/best.pt" --examples benchmarks/serbian_25.jsonl \
    --output "$output/benchmark_$protocol.json" --question-format "$protocol" \
    --max-new-tokens 32 --device cpu --precision fp32
done
