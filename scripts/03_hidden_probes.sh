#!/bin/bash
# Ablation: probes on residual-stream hidden states instead of attention heads.
# Run after 02_indomain.sh (it reuses the translated test sets).
# Usage: bash scripts/03_hidden_probes.sh [seed ...]     (default: 0 1 2 3 4)
set -euo pipefail
source "$(dirname "$0")/common.sh"
if [ $# -gt 0 ]; then SEEDS=("$@"); fi

for n in "${SEEDS[@]}"; do
    for d in "${TRAIN_DATASETS[@]}"; do
        for m in "${PROBE_MODELS[@]}"; do
            python -u main_detection_probes_hidden.py \
                --dataset "$d" --train train --test test --exp_num "$n" \
                --model_name "$m" --do_probes --do_translation
        done
    done
done
