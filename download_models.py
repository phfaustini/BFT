"""Pre-download every model used in the paper into the Hugging Face cache.

Useful on clusters whose compute nodes have no internet access: run this on a
login node, then launch the jobs with TRANSFORMERS_OFFLINE=1 / HF_HUB_OFFLINE=1.

    hf auth login            # Llama models are gated
    python download_models.py

Set HF_HOME to control where the cache lives.
"""
from huggingface_hub import snapshot_download


MODELS = [
    # Probed LLMs (attention-head / hidden-state probes) and FJD backbones
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen3-8B",
    # Embedding baselines (also used for greedy dataset construction)
    "sentence-transformers/all-mpnet-base-v2",
    "Qwen/Qwen3-Embedding-8B",
    # Perplexity baseline
    "EleutherAI/gpt-neo-2.7B",
    # Off-the-shelf guards
    "meta-llama/Llama-Guard-3-8B",
    "meta-llama/Llama-Prompt-Guard-2-86M",
    "MickyMike/DecipherGuard",
    # Translation models for the cross-lingual evaluation
    "Helsinki-NLP/opus-mt-tc-big-en-es",
    "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "Helsinki-NLP/opus-mt-en-zh",
    "Helsinki-NLP/opus-mt-synthetic-en-so",
    "Helsinki-NLP/opus-mt-synthetic-en-mk",
]


if __name__ == "__main__":
    for repo_id in MODELS:
        print(f"Downloading {repo_id} ...")
        snapshot_download(repo_id)
