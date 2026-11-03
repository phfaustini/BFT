import json
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI

from chatgpt_dispatch import MANIFEST_PATH, WORKDIR, _load_manifest


def collect_chatgpt_batch(batch_id: str, n_prompts: int) -> list[str | None]:
    """Fetch results for a completed batch. Returns answers aligned to input order."""
    client = OpenAI()  # reads OPENAI_API_KEY
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch_id} status is {batch.status}, not completed")

    results: list[str | None] = [None] * n_prompts
    output = client.files.content(batch.output_file_id).text
    for line in output.splitlines():
        record = json.loads(line)
        idx = int(record["custom_id"].split("-")[1])
        if record.get("error"):
            continue
        results[idx] = record["response"]["body"]["choices"][0]["message"]["content"]
    return results


if __name__ == "__main__":
    manifest = _load_manifest()
    if not manifest:
        raise SystemExit(f"No manifest found at {MANIFEST_PATH}")

    client = OpenAI()  # reads OPENAI_API_KEY
    for dataset_name, entry in manifest.items():
        batch = client.batches.retrieve(entry["batch_id"])
        print(f"{dataset_name}: {batch.status} "
              f"({batch.request_counts.completed}/{batch.request_counts.total})")
        if batch.status != "completed":
            continue

        prompts = pd.read_parquet(entry["prompts_path"])
        answers = collect_chatgpt_batch(entry["batch_id"], len(prompts))
        prompts = prompts.assign(chatgpt_answer=answers)
        out_path = Path(f"batch_jobs/chatgpt_answers_{dataset_name}.parquet")
        prompts.to_parquet(out_path)
        print(f"  wrote {out_path}")
