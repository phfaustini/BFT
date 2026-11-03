"""Download and materialise the base datasets used in the paper under ./data.

Run from the repository root:

    python generate_datasets.py

WildGuardMix and WildJailbreak are gated on the Hugging Face Hub: accept their
terms on the dataset pages and run `hf auth login` first. Set HF_HOME to
control where the Hugging Face cache lives.

The two attack-derived datasets (ReNeLLM + WildGuardMix and AutoDAN-Turbo +
WildGuardMix) need Qwen3-Embedding-8B embeddings of WildGuardMix; they are
built afterwards by build_greedy_datasets.py (see README).
"""
import os

from datasets import load_dataset

from detection.models_datasets import get_dataset


def materialise_renellm() -> None:
    """Write data/ReNeLLM-Jailbreak/renellm_train.csv exactly as the loader does.

    get_dataset("renellm") only reads its DataFrame from this CSV, so the CSV
    has to exist before the loader is called on a fresh clone.
    """
    path = "data/ReNeLLM-Jailbreak/renellm_train.csv"
    if not os.path.exists(path):
        ds = load_dataset("Deep1994/ReNeLLM-Jailbreak")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ds["train"].to_pandas().to_csv(path, index=False)


# (dataset name, splits) as passed to the experiment entrypoints.
BASE_DATASETS = [
    ("TrustAIRLab", ["train", "test"]),
    ("allenai_wildjailbreak", ["train", "test"]),
    ("vanillawildjailbreak", ["train", "test"]),
    ("wildguardmix", ["train", "test"]),
    # Raw attack corpora; only used to build the greedy datasets and for the
    # GPT-5.4-mini experiments (Appendix A.7).
    ("renellm", ["train"]),
    ("autodanturbo", ["test"]),
]


if __name__ == "__main__":
    materialise_renellm()
    for name, splits in BASE_DATASETS:
        for split in splits:
            df = get_dataset(name, split)
            n_mal = int(df["is_malicious"].sum()) if "is_malicious" in df else -1
            print(f"{name:25s} {split:6s} rows={len(df):6d} malicious={n_mal}")
