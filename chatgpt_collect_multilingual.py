"""Collect GPT-5.4-mini answers for the *translated* batch jobs.

Companion to chatgpt_collect.py, run after chatgpt_dispatch_multilingual.py.
The multilingual dispatch records its batches in the same manifest
(batch_jobs/manifest.json) but tags them "<dataset>_<lang>" and stores extra
"language" / "dataset" keys. This script processes exactly those entries,
fetches the completed batches and writes one answers file per (dataset, lang).

Outputs, per multilingual entry:
    batch_jobs/chatgpt_answers_<dataset>_<lang>.parquet

Needs OPENAI_API_KEY in the environment.
"""
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI

from chatgpt_collect import collect_chatgpt_batch
from chatgpt_dispatch import MANIFEST_PATH, WORKDIR, _load_manifest


if __name__ == "__main__":
    manifest = _load_manifest()
    if not manifest:
        raise SystemExit(f"No manifest found at {MANIFEST_PATH}")

    # Multilingual entries are the ones tagged with a language by the
    # multilingual dispatch; plain English runs lack the "language" key.
    multilingual = {tag: entry for tag, entry in manifest.items()
                    if "language" in entry}
    if not multilingual:
        raise SystemExit("No multilingual entries found in manifest "
                         "(run chatgpt_dispatch_multilingual.py --submit first)")

    client = OpenAI()  # reads OPENAI_API_KEY
    for tag, entry in multilingual.items():
        batch = client.batches.retrieve(entry["batch_id"])
        print(f"{tag}: {batch.status} "
              f"({batch.request_counts.completed}/{batch.request_counts.total})")
        if batch.status != "completed":
            continue

        prompts = pd.read_parquet(entry["prompts_path"])
        answers = collect_chatgpt_batch(entry["batch_id"], len(prompts))
        prompts = prompts.assign(chatgpt_answer=answers)
        out_path = WORKDIR / f"chatgpt_answers_{tag}.parquet"
        prompts.to_parquet(out_path)
        print(f"  wrote {out_path}")
