#!/bin/bash
# Submit one stage of the paper's experiment matrix as PBS jobs.
# Wait for each stage to finish before submitting the next one.
#
#   python generate_datasets.py                    # on a node with internet access
#   bash scripts/pbs/submit_all.sh greedy          # build the two greedy datasets
#   bash scripts/pbs/submit_all.sh embeddings      # then: python stage_embeddings.py
#   bash scripts/pbs/submit_all.sh indomain
#   bash scripts/pbs/submit_all.sh hidden
#   bash scripts/pbs/submit_all.sh ood
set -euo pipefail
source "$(dirname "$0")/../common.sh"   # also cd's to the repository root
P=scripts/pbs

# Short job names (some PBS installations limit names to 15 characters).
short() {
    case "$1" in
        wildguardmix) echo WG ;;
        vanillawildjailbreak) echo VWJ ;;
        TrustAIRLab) echo TAIR ;;
        allenai_wildjailbreak) echo WJB ;;
        greedy_renellm_wildguardmix) echo RWG ;;
        greedy_autodanturbo_wildguardmix) echo AWG ;;
        *) echo "$1" | tr -cd 'A-Za-z0-9' | cut -c1-6 ;;
    esac
}
mshort() { case "$1" in *Llama*) echo l ;; *Qwen*) echo q ;; *) echo m ;; esac; }

case "${1:-}" in
    greedy)
        qsub -N greedy "$P/build_greedy_datasets.qsub" ;;
    embeddings)
        for d in "${OOD_TEST_DATASETS[@]}"; do
            qsub -N "emb_$(short "$d")" -v d="$d" "$P/qwen_embeddings.qsub"
        done ;;
    indomain)
        for n in "${SEEDS[@]}"; do
            for d in "${TRAIN_DATASETS[@]}"; do
                qsub -N "bl_$(short "$d")$n" -v n="$n",d="$d" "$P/indomain_baselines.qsub"
                for m in "${PROBE_MODELS[@]}"; do
                    qsub -N "pr$(mshort "$m")_$(short "$d")$n" -v n="$n",d="$d",m="$m" "$P/indomain_probes.qsub"
                done
            done
            for d in "${TEST_ONLY_DATASETS[@]}"; do
                qsub -N "bl_$(short "$d")$n" -v n="$n",d="$d",zeroshot=yes "$P/indomain_baselines.qsub"
            done
        done ;;
    hidden)
        for n in "${SEEDS[@]}"; do
            for d in "${TRAIN_DATASETS[@]}"; do
                for m in "${PROBE_MODELS[@]}"; do
                    qsub -N "hp$(mshort "$m")_$(short "$d")$n" -v n="$n",d="$d",m="$m" "$P/hidden_probes.qsub"
                done
            done
        done ;;
    ood)
        n=0
        for trd in "${TRAIN_DATASETS[@]}"; do
            for ted in "${OOD_TEST_DATASETS[@]}"; do
                [ "$trd" = "$ted" ] && continue
                case "$trd:$ted" in
                    vanillawildjailbreak:allenai_wildjailbreak|allenai_wildjailbreak:vanillawildjailbreak) continue ;;
                esac
                qsub -N "ood_$(short "$trd")2$(short "$ted")" -v n="$n",trd="$trd",ted="$ted" "$P/out_of_domain.qsub"
            done
        done ;;
    *)
        echo "usage: $0 {greedy|embeddings|indomain|hidden|ood}" >&2; exit 1 ;;
esac
