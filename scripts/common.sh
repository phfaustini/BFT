# Shared settings for the run scripts. Sourced, not executed.

# All entrypoints use paths relative to the repository root.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1
# Uncomment on machines without internet access (run download_models.py first):
# export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

# Seeds (experiment numbers) used in the paper.
SEEDS=(0 1 2 3 4)

# Datasets with a training set (in-domain runs, OOD training sets).
TRAIN_DATASETS=(wildguardmix vanillawildjailbreak TrustAIRLab allenai_wildjailbreak greedy_renellm_wildguardmix)

# Test-only dataset: evaluated with the off-the-shelf detectors in-domain and as an OOD target.
TEST_ONLY_DATASETS=(greedy_autodanturbo_wildguardmix)

# Out-of-domain targets.
OOD_TEST_DATASETS=("${TRAIN_DATASETS[@]}" "${TEST_ONLY_DATASETS[@]}")

# LLMs whose activations are probed.
PROBE_MODELS=("meta-llama/Llama-3.2-1B-Instruct" "Qwen/Qwen3-8B")
