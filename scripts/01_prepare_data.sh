#!/bin/bash
# Build all datasets, pre-compute Qwen3-Embedding-8B embeddings and translated
# test sets (es, pt, zh, so, mk), and stage them into detection/results.
# Needs a GPU. Usage: bash scripts/01_prepare_data.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"

python -u generate_datasets.py
python -u build_greedy_datasets.py

for d in "${OOD_TEST_DATASETS[@]}"; do
    PYTHONHASHSEED=42 python -u qwen_embeddings.py --dataset "$d" --train train --test test
done

python -u stage_embeddings.py --exp_nums "${SEEDS[@]}"
