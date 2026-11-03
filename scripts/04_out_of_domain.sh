#!/bin/bash
# Out-of-domain experiments: detectors trained in-domain (02_indomain.sh) on one
# dataset are evaluated on another. Pairs: every training set x every target,
# except a dataset with itself and VanillaWildJailbreak <-> WildJailbreak
# (VanillaWildJailbreak is a subset of WildJailbreak).
# Usage: bash scripts/04_out_of_domain.sh [seed]     (default: 0, as in the paper)
set -euo pipefail
source "$(dirname "$0")/common.sh"
n="${1:-0}"

for trd in "${TRAIN_DATASETS[@]}"; do
    for ted in "${OOD_TEST_DATASETS[@]}"; do
        [ "$trd" = "$ted" ] && continue
        case "$trd:$ted" in
            vanillawildjailbreak:allenai_wildjailbreak|allenai_wildjailbreak:vanillawildjailbreak) continue ;;
        esac
        common_args=(--exp_num "$n"
                     --train_dataset "$trd" --train_split train --train_indomain_test_split test
                     --test_dataset "$ted" --test_split test --test_indomain_train_split train)
        python -u main_detection_out_of_domain.py "${common_args[@]}" --do_tfidf --do_embeddings
        for m in "${PROBE_MODELS[@]}"; do
            python -u main_detection_out_of_domain.py "${common_args[@]}" --model_name "$m" --do_probes
        done
    done
done
