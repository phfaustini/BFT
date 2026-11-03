import json
import os
import time
from pathlib import Path

from openai import OpenAI

import detection.models_datasets as md


WORKDIR = Path("batch_jobs")
MANIFEST_PATH = WORKDIR / "manifest.json"


def dispatch_chatgpt_batch(
    prompts: list[str],
    tag: str,
    model: str = "gpt-5.4-mini",
    system_prompt: str | None = None,
) -> str:
    """Submit `prompts` as a single OpenAI batch job. Returns the batch id."""
    client = OpenAI()  # reads OPENAI_API_KEY
    WORKDIR.mkdir(parents=True, exist_ok=True)

    input_path = WORKDIR / f"batch_input_{tag}_{int(time.time())}.jsonl"
    with input_path.open("w") as f:
        for i, prompt in enumerate(prompts):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            request = {
                "custom_id": f"req-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": model, "messages": messages},
            }
            f.write(json.dumps(request) + "\n")

    with input_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"tag": tag},
    )
    return batch.id


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    DATASETS = ["wildguardmix",
                "TrustAIRLab",
                "allenai_wildjailbreak",
                "greedy_renellm_wildguardmix",
                "autodanturbo"]

    manifest = _load_manifest()
    for dataset_name in DATASETS:
        df_test = md.get_dataset(dataset_name, "test")
        df_test = df_test[df_test['is_malicious'] == 1].copy()
        prompts = df_test.sample(100, random_state=42).reset_index(drop=True)

        WORKDIR.mkdir(parents=True, exist_ok=True)
        prompts_path = WORKDIR / f"prompts_{dataset_name}.csv"
        prompts.to_csv(prompts_path)
        prompts_path = WORKDIR / f"prompts_{dataset_name}.parquet"
        prompts.to_parquet(prompts_path)        

        batch_id = dispatch_chatgpt_batch(prompts['query'].tolist(), tag=dataset_name)
        manifest[dataset_name] = {
            "batch_id": batch_id,
            "prompts_path": str(prompts_path),
        }
        _save_manifest(manifest)
        print(f"Submitted {dataset_name}: {batch_id}")
