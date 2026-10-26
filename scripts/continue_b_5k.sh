#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON:-python}"
output=runs/deep_continue_B_5k

"$python_bin" -u -m serbian_llm continue-train \
  --config configs/continued_deep_b_5k.json \
  --init-from runs/deep_continue_B/best.pt \
  --tokenizer data/serbian_expanded_v3/bpe8k \
  --output "$output" --device cuda:0

"$python_bin" -u -m serbian_llm evaluate \
  --checkpoint "$output/best.pt" \
  --data data/serbian_mix_v2/corpus/validation.jsonl \
  --output "$output/full_validation.json" \
  --limit 0 --stride 256 --device cuda:0 --precision bf16

"$python_bin" -u -m serbian_llm benchmark \
  --checkpoint "$output/best.pt" \
  --examples benchmarks/serbian_25.jsonl \
  --output "$output/benchmark_cpu.json" \
  --max-new-tokens 32 --device cpu --precision fp32
