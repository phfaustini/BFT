"""Dispatch GPT-5.4-mini jailbreak experiments on *translated* prompts.

Companion to chatgpt_dispatch.py. To test whether multilingual prompts are a
heightened jailbreak risk, we take the same 100 malicious prompts that were
sent to GPT-5.4-mini in English (batch_jobs/prompts_<dataset>.parquet) and
re-send their translations into Spanish, Portuguese, Chinese, Somali and
Macedonian. Re-using the English sample makes this a paired comparison: any
change in answer rate is due to language, not to which prompts were drawn,
and it extends the existing English results (Table~tab:judge) directly.

Translated prompts come pre-computed from embeddings/
df_<dataset>_test_<lang>.csv (columns: query = translated, query_en =
original English, is_malicious). We match the English sample by query_en.

Outputs, per (dataset, language):
    batch_jobs/prompts_<dataset>_<lang>.{parquet,csv}   the dispatched prompts
    an entry in batch_jobs/manifest.json with the batch id

Safety: this submits paid OpenAI batch jobs, so it is a DRY RUN by default
(writes the prompt files, prints what it would submit). Pass --submit to
actually dispatch. Needs OPENAI_API_KEY in the environment.

    python chatgpt_dispatch_multilingual.py            # dry run
    python chatgpt_dispatch_multilingual.py --submit    # actually dispatch
"""
import argparse
from pathlib import Path

import pandas as pd

from chatgpt_dispatch import (
    WORKDIR, dispatch_chatgpt_batch, _load_manifest, _save_manifest,
)

EMB = Path("embeddings")  # translated test sets written by qwen_embeddings.py
LANGS = ["spanish", "portuguese", "chinese", "somali", "macedonian"]
# Datasets dispatched in English (chatgpt_dispatch.py) that also have
# translations and whose English sample pairs cleanly by query_en. These are
# exactly the datasets in the cross-lingual results. (autodanturbo_wildguardmix
# is excluded: its English sample does not match the translated query_en and it
# lacks Somali/Macedonian translations.)
DATASETS = ["wildguardmix", "TrustAIRLab", "allenai_wildjailbreak",
            "greedy_renellm_wildguardmix"]
N = 100


def english_sample(dataset: str) -> list[str] | None:
    """The exact English queries dispatched for `dataset`, or None if absent."""
    p = WORKDIR / f"prompts_{dataset}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)["query"].astype(str).str.strip().tolist()


def select_translated(dataset: str, lang: str,
                      eng: list[str] | None) -> pd.DataFrame | None:
    """Translated malicious prompts for `dataset`/`lang`, paired to the English
    sample by query_en (falls back to a fresh 100-sample if no English run)."""
    csv = EMB / f"df_{dataset}_test_{lang}.csv"
    if not csv.exists():
        print(f"  [skip] {csv} not found")
        return None
    df = pd.read_csv(csv)
    df = df[df["is_malicious"] == 1].copy()
    df["query_en"] = df["query_en"].astype(str).str.strip()
    df = df.drop_duplicates("query_en")
    if eng is not None:
        order = {q: i for i, q in enumerate(eng)}
        sub = df[df["query_en"].isin(order)].copy()
        sub = sub.sort_values("query_en", key=lambda s: s.map(order)).reset_index(drop=True)
        if len(sub) < len(eng):
            print(f"  [warn] {dataset}/{lang}: matched {len(sub)}/{len(eng)} "
                  f"English prompts by query_en")
        return sub
    sub = df.sample(min(N, len(df)), random_state=0).reset_index(drop=True)
    print(f"  [info] {dataset}/{lang}: no English run found, fresh sample n={len(sub)}")
    return sub


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submit", action="store_true",
                    help="actually submit the OpenAI batch jobs (default: dry run)")
    ap.add_argument("--model", default="gpt-5.4-mini")
    args = ap.parse_args()

    manifest = _load_manifest()
    for dataset in DATASETS:
        eng = english_sample(dataset)
        if eng is None:
            print(f"[{dataset}] no English prompts file; will sample fresh per language")
        for lang in LANGS:
            sub = select_translated(dataset, lang, eng)
            if sub is None or sub.empty:
                continue
            tag = f"{dataset}_{lang}"
            WORKDIR.mkdir(parents=True, exist_ok=True)
            sub.to_parquet(WORKDIR / f"prompts_{tag}.parquet")
            sub.to_csv(WORKDIR / f"prompts_{tag}.csv", index=False)

            if not args.submit:
                print(f"[dry-run] {tag}: would submit {len(sub)} prompts")
                continue
            batch_id = dispatch_chatgpt_batch(sub["query"].tolist(), tag=tag,
                                              model=args.model)
            manifest[tag] = {
                "batch_id": batch_id,
                "prompts_path": str(WORKDIR / f"prompts_{tag}.parquet"),
                "language": lang,
                "dataset": dataset,
            }
            _save_manifest(manifest)
            print(f"Submitted {tag}: {batch_id}")

    if not args.submit:
        print("\nDry run only. Re-run with --submit to dispatch the batch jobs.")


if __name__ == "__main__":
    main()
