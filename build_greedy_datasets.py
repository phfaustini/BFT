"""Build the two attack-derived datasets of the paper (Appendix A.1).

* greedy_renellm_wildguardmix (ReNe+WG): ReNeLLM rewrites of AdvBench as the
  malicious class, WildGuardMix benign prompts as the benign class. ReNeLLM is
  split 50:50 by original harmful behaviour (group-aware, so no seed leaks
  across splits); each malicious WildGuardMix test prompt is then greedily
  matched 1:1 to its most cosine-similar unused ReNeLLM test-pool prompt in the
  Qwen3-Embedding-8B space.
* greedy_autodanturbo_wildguardmix (AutoDAN-Turbo+WG, test only): AutoDAN-Turbo
  lifelong-attack prompts (score >= 8.5) paired 1:1 with their most similar
  WildGuardMix benign test prompts. The construction lives in
  detection/models_datasets.py and is triggered here.

Prerequisites: python generate_datasets.py. English Qwen3-Embedding-8B
embeddings of WildGuardMix and ReNeLLM are read from ./embeddings and computed
(on GPU) when missing.

Outputs:
    data/combined_greedy_renellm_wildguardmix/combined.csv
    data/combined_greedy_autodanturbo_wildguardmix/combined.csv
    embeddings/embeddings_greedy_*_{train,test}_eng.npy

Run from the repository root:

    python build_greedy_datasets.py
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random

import numpy as np
import pandas as pd
import torch

from detection.models_datasets import get_dataset
from detection.detection_baselines import convert_text_to_embeddings


EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_deterministic():
    """Same settings as qwen_embeddings.py."""
    if 'cuda' in DEVICE.type:
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)


def english_embeddings(dataset_name: str, split_tag: str, df: pd.DataFrame) -> np.ndarray:
    """Load embeddings/embeddings_<dataset>_<split_tag>_eng.npy, computing it if missing
    (same model and settings as qwen_embeddings.py, English only)."""
    path = f"embeddings/embeddings_{dataset_name}_{split_tag}_eng.npy"
    if not os.path.exists(path):
        print(f"Computing {path} ({len(df)} queries) ...")
        set_deterministic()
        emb = convert_text_to_embeddings(df['query'].to_list(), EMBEDDING_MODEL)
        os.makedirs("embeddings", exist_ok=True)
        with open(path, 'wb') as f:
            np.save(f, emb.cpu().float().numpy())
    with open(path, 'rb') as f:
        return np.load(f)


def build_greedy_renellm_wildguardmix() -> None:
    out_csv = "data/combined_greedy_renellm_wildguardmix/combined.csv"
    if os.path.exists(out_csv):
        print(f"{out_csv} exists, skipping.")
        return

    df_re = get_dataset("renellm", "test")  # Only malicious (the loader returns the full set)
    df_wg = get_dataset("wildguardmix", "test")  # Benign and malicious
    df_wg_train = get_dataset("wildguardmix", "train")  # Benign + malicious; only the benign side is used.

    embeddings_re = english_embeddings("renellm", "train", df_re)
    embeddings_wg = english_embeddings("wildguardmix", "test", df_wg)
    embeddings_wg_train = english_embeddings("wildguardmix", "train", df_wg_train)

    assert df_re.shape[0] == embeddings_re.shape[0]
    assert df_wg.shape[0] == embeddings_wg.shape[0]
    assert df_wg_train.shape[0] == embeddings_wg_train.shape[0]

    df_re["embedding"] = list(embeddings_re)
    df_wg["embedding"] = list(embeddings_wg)
    df_wg_train["embedding"] = list(embeddings_wg_train)

    df_wg_b_test = df_wg[df_wg['is_malicious'] == 0].reset_index(drop=True)
    df_wg_m_test = df_wg[df_wg['is_malicious'] == 1].reset_index(drop=True)
    df_wg_b_train = df_wg_train[df_wg_train['is_malicious'] == 0].reset_index(drop=True)

    embeddings_wg_m = np.stack(df_wg_m_test["embedding"].values)

    # Split renellm 50:50 by original_harm_behavior (group-aware) so no original leaks across splits.
    # All variants of a given original go to the same side.
    df_re = df_re.reset_index(drop=True)
    unique_originals = df_re["original_harm_behavior"].unique()
    shuffled_originals = np.random.default_rng(seed=42).permutation(unique_originals)
    half = len(shuffled_originals) // 2
    test_originals = set(shuffled_originals[:half])
    train_originals = set(shuffled_originals[half:])
    assert test_originals.isdisjoint(train_originals)

    re_test_mask = df_re["original_harm_behavior"].isin(test_originals).values
    re_train_mask = df_re["original_harm_behavior"].isin(train_originals).values

    # Positional indices into df_re / embeddings_re (kept aligned).
    re_test_pos = np.where(re_test_mask)[0]
    re_train_pos = np.where(re_train_mask)[0]
    df_re_test_pool = df_re.iloc[re_test_pos].reset_index(drop=True)
    df_re_train_pool = df_re.iloc[re_train_pos].reset_index(drop=True)
    embeddings_re_test_pool = embeddings_re[re_test_pos]

    # Cosine similarity: every renellm test-pool row vs every malicious wg test row.
    re_norm = embeddings_re_test_pool / np.linalg.norm(embeddings_re_test_pool, axis=1, keepdims=True)
    wgm_norm = embeddings_wg_m / np.linalg.norm(embeddings_wg_m, axis=1, keepdims=True)
    cos_sim = re_norm @ wgm_norm.T  # shape: (n_re_test_pool, n_wg_m)

    # Greedy matching within the test pool: for each malicious wg row, pick the closest unused
    # renellm test-pool row.
    n_re_pool, n_wg_m = cos_sim.shape
    assert n_wg_m <= n_re_pool, "Not enough test-pool rows to do 1:1 matching."
    picked_pool_idx = np.empty(n_wg_m, dtype=np.int64)
    used = np.zeros(n_re_pool, dtype=bool)
    for j in range(n_wg_m):
        sims = cos_sim[:, j].copy()
        sims[used] = -np.inf
        idx = int(np.argmax(sims))
        picked_pool_idx[j] = idx
        used[idx] = True

    matched_sims = cos_sim[picked_pool_idx, np.arange(n_wg_m)]

    df_re_matched = df_re_test_pool.iloc[picked_pool_idx].reset_index(drop=True)
    # Each re-matched row j was picked because it was closest to df_wg_m_test row j.
    df_re_matched["closest_query"] = df_wg_m_test["query"].values
    df_re_matched["cos_sim"] = matched_sims

    # Sanity: by construction, every matched original belongs to test_originals.
    matched_originals = set(df_re_matched["original_harm_behavior"].unique())
    assert matched_originals.issubset(test_originals)
    assert matched_originals.isdisjoint(train_originals)

    # Sample malicious training rows from the leakage-free renellm train pool, sized to the
    # wg benign train count so the final train set is class-balanced.
    n_train_target = min(df_re_train_pool.shape[0], df_wg_b_train.shape[0])
    sampled_train_pos = np.random.default_rng(seed=42).choice(
        df_re_train_pool.shape[0], size=n_train_target, replace=False,
    )
    df_re_train_sampled = df_re_train_pool.iloc[sampled_train_pos].reset_index(drop=True)

    # Final train/test assembly. Benign rows come from wildguardmix; malicious rows from renellm.
    KEEP_COLS = ["query", "target", "is_malicious", "embedding", "closest_query", "cos_sim"]

    def _slice(df, source):
        # reindex tolerates rows that weren't matched (wg-benign + re-train-sampled):
        # closest_query / cos_sim become NaN for them.
        out = df.reindex(columns=KEEP_COLS).copy()
        out["source"] = source
        return out

    # Subsample wg benign test down to the malicious count so the test set is class-balanced.
    n_test_each = df_re_matched.shape[0]
    assert n_test_each <= df_wg_b_test.shape[0], "Not enough wg benign test rows to balance."
    wg_b_test_pos = np.random.default_rng(seed=42).choice(
        df_wg_b_test.shape[0], size=n_test_each, replace=False,
    )
    df_wg_b_test_sampled = df_wg_b_test.iloc[wg_b_test_pos].reset_index(drop=True)

    df_test = pd.concat(
        [_slice(df_wg_b_test_sampled, "wildguardmix"),
         _slice(df_re_matched, "renellm")],
        ignore_index=True,
    )
    df_test["split"] = "test"

    df_train = pd.concat(
        [_slice(df_wg_b_train, "wildguardmix"),
         _slice(df_re_train_sampled, "renellm")],
        ignore_index=True,
    )
    df_train["split"] = "train"

    # Final leakage check across the two splits.
    test_re_originals = set(df_re_matched["original_harm_behavior"].unique())
    train_re_originals = set(df_re_train_sampled["original_harm_behavior"].unique())
    assert test_re_originals.isdisjoint(train_re_originals), "renellm original leaked across splits"

    print(f"greedy_renellm_wildguardmix train: {df_train.shape}  "
          f"benign={int((df_train['is_malicious']==0).sum())}  malicious={int((df_train['is_malicious']==1).sum())}")
    print(f"greedy_renellm_wildguardmix test:  {df_test.shape}  "
          f"benign={int((df_test['is_malicious']==0).sum())}  malicious={int((df_test['is_malicious']==1).sum())}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.concat([df_train, df_test], ignore_index=True)
    embeddings_train = np.stack(df_train['embedding'].copy().values)
    embeddings_test = np.stack(df_test['embedding'].copy().values)
    del df['embedding']
    df.to_csv(out_csv, index=False)
    with open("embeddings/embeddings_greedy_renellm_wildguardmix_test_eng.npy", 'wb') as f:
        np.save(f, embeddings_test)
    with open("embeddings/embeddings_greedy_renellm_wildguardmix_train_eng.npy", 'wb') as f:
        np.save(f, embeddings_train)
    del df_train['embedding']
    del df_test['embedding']
    df_train.to_csv('embeddings/df_greedy_renellm_wildguardmix_train_eng.csv', index=False)
    df_test.to_csv('embeddings/df_greedy_renellm_wildguardmix_test_eng.csv', index=False)


def build_greedy_autodanturbo_wildguardmix() -> None:
    # The loader reads embeddings/embeddings_wildguardmix_test_eng.npy, so make sure it exists.
    english_embeddings("wildguardmix", "test", get_dataset("wildguardmix", "test"))
    df = get_dataset("greedy_autodanturbo_wildguardmix", "test")
    print(f"greedy_autodanturbo_wildguardmix test: {df.shape}  "
          f"benign={int((df['is_malicious']==0).sum())}  malicious={int((df['is_malicious']==1).sum())}")


if __name__ == "__main__":
    build_greedy_renellm_wildguardmix()
    build_greedy_autodanturbo_wildguardmix()
