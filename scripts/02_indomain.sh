#!/bin/bash
# In-domain and cross-lingual experiments (baselines + attention-head probes).
# Usage: bash scripts/02_indomain.sh [seed ...]     (default: 0 1 2 3 4)
set -euo pipefail
source "$(dirname "$0")/common.sh"
if [ $# -gt 0 ]; then SEEDS=("$@"); fi

for n in "${SEEDS[@]}"; do
    for d in "${TRAIN_DATASETS[@]}"; do
        # TF-IDF, embeddings (MPNet, Qwen3-Emb), perplexity, FJD, guard models
        PYTHONHASHSEED=42 python -u main_detection.py \
            --dataset "$d" --train train --test test --exp_num "$n" \
            --do_tfidf --do_embeddings --do_ppl --do_fjd --do_llamaguard --do_translation
        # Attention-head probes, top-k in {1,3,5,10,15,20,30,50}
        for m in "${PROBE_MODELS[@]}"; do
            PYTHONHASHSEED=42 python -u main_detection.py \
                --dataset "$d" --train train --test test --exp_num "$n" \
                --model_name "$m" --do_probes --do_translation
        done
    done
    # Test-only dataset: off-the-shelf detectors only
    for d in "${TEST_ONLY_DATASETS[@]}"; do
        PYTHONHASHSEED=42 python -u main_detection.py \
            --dataset "$d" --train train --test test --exp_num "$n" \
            --do_ppl --do_fjd --do_llamaguard --do_translation
    done
done
