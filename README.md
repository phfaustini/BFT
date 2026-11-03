# BFT: Before the First Token

Code for the EMNLP 2026 paper **"Before the First Token: Jailbreak Detection with Attention Probes"**.

> Large language models (LLMs) are deployed with safety guardrails, yet jailbreak attacks and malicious prompts keep bypassing them. Re-aligning an LLM each time a new attack appears is more expensive than crafting attacks, so jailbreak detection is increasingly framed as a pre-processing step: a cheap detector filters out a malicious prompt before it reaches the LLM. This need is reinforced by multilingual attacks, where simply translating a prompt to another language can bypass the guardrails. Our goal is to find whether internal representations of a frozen LLM already separate malicious from benign prompts, and whether such a signal is resilient against translation. We study two types of detectors that require no fine-tuning of the LLM: a logistic-regression classifier over whole-sentence embeddings, and a probe-based detector that trains one logistic-regression probe per attention head on the attention activations of the last input token. They need at most a single forward pass. Our experiments show that both match or outperform strong sentence-embedding detectors and off-the-shelf baselines, specially under translation. The probes degrade more gracefully on higher-resource languages and remain competitive on lower-resource ones. Our results indicate that a frozen LLM linearly encodes whether an incoming prompt is malicious in a small set of mid-to-late layers. Per-head attention probes thus offer a defence that can be cheaply re-trained as new attacks emerge, without touching the underlying model.

This repository contains the code to build the datasets and run every experiment in the paper. It does **not** include datasets, trained detectors, cached embeddings due to size and some licence constraints.


## Citation

```bibtex
@inproceedings{faustini-etal-2026-first,
    title = "Before the First Token: Jailbreak Detection with Attention Probes",
    author = "Faustini, Pedro  and
      McIver, Annabelle  and
      Dras, Mark",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-emnlp.2242/",
    doi = "10.18653/2026.findings-emnlp.2242",
    pages = "48011--48029",
    ISBN = "979-8-89176-462-0",
}
```


## Methods

| Detector | Code | Notes |
|---|---|---|
| **Attention-head probes** (ours) | [detection/probes.py](detection/probes.py) | Last-token output of every attention head (hook on `self_attn.o_proj`) of a frozen 4-bit LLM; one StandardScaler + LogisticRegression per head, heads ranked by validation F1, top-*k* ensembled. LLMs: Llama-3.2-1B-Instruct, Qwen3-8B. *k* ∈ {1, 3, 5, 10, 15, 20, 30, 50}. |
| Hidden-state probes (ablation) | [detection/probes_hidden.py](detection/probes_hidden.py) | Same, on per-layer residual-stream hidden states. |
| Embeddings + LR | [detection/detection_baselines.py](detection/detection_baselines.py) | all-mpnet-base-v2 and Qwen3-Embedding-8B. |
| TF-IDF + LR | [detection/detection_baselines.py](detection/detection_baselines.py) | |
| Perplexity | [detection/detection_baselines.py](detection/detection_baselines.py) | GPT-Neo-2.7B. |
| FJD | [detection/detection_baselines.py](detection/detection_baselines.py) | First-token jailbreak detection on Qwen3-8B and Llama-3.2-1B-Instruct. |
| Guard models | [detection/detection_baselines.py](detection/detection_baselines.py) | Llama-Guard-3-8B, Llama-Prompt-Guard-2-86M, DecipherGuard (zero-shot). |


## Datasets

| Name used in the code | Paper | Source |
|---|---|---|
| `wildguardmix` | WildGuardMix (WG) | [allenai/wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix) (gated) |
| `TrustAIRLab` | TrustAIR In-The-Wild Jailbreak | [TrustAIRLab/in-the-wild-jailbreak-prompts](https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts) |
| `allenai_wildjailbreak` | WildJailbreak | [allenai/wildjailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) (gated) |
| `vanillawildjailbreak` | VanillaWildJailbreak | vanilla subset of WildJailbreak |
| `greedy_renellm_wildguardmix` | ReNe+WG | [ReNeLLM-Jailbreak](https://huggingface.co/datasets/Deep1994/ReNeLLM-Jailbreak) + WildGuardMix benign prompts, greedy embedding matching |
| `greedy_autodanturbo_wildguardmix` | AutoDAN-Turbo+WG (test only) | [AutoDAN-Turbo](https://github.com/SaFo-Lab/AutoDAN-Turbo) lifelong-attack log + WildGuardMix benign prompts, greedy embedding matching |

All loaders live in [detection/models_datasets.py](detection/models_datasets.py) and return a DataFrame with a `query` text column and an `is_malicious ∈ {0, 1}` label. The loader also contains datasets explored during development that are not used in the paper (e.g. BeaverTails, AdvBench); they are kept for completeness.

Datasets are downloaded from their original sources and are not redistributed here. Please check each dataset's licence and terms of use.

## Setup

```bash
conda env create -f environment.yml
conda activate bft
```

Several models and datasets are gated on the Hugging Face Hub. Accept their terms on the Hub pages (Llama-3.2, Llama-Guard-3, Prompt-Guard-2, WildGuardMix, WildJailbreak) and log in:

```bash
hf auth login
```

Set `HF_HOME` if you want the Hugging Face cache somewhere other than `~/.cache/huggingface`. On machines whose compute nodes have no internet access, pre-download all models with `python download_models.py` and uncomment the offline variables in [scripts/common.sh](scripts/common.sh).

The original experiments ran on single NVIDIA H100 GPUs with 128 GB of RAM. LLMs are loaded in 4-bit NF4 quantisation.

## Running the experiments

All commands are run from the repository root. Outputs go to `data/`, `embeddings/` and `detection/results/`.

### 1. Data preparation

```bash
bash scripts/01_prepare_data.sh
```

This runs, in order:

1. [generate_datasets.py](generate_datasets.py): downloads and materialises the base datasets under `data/`.
2. [build_greedy_datasets.py](build_greedy_datasets.py): builds ReNe+WG and AutoDAN-Turbo+WG by greedy 1:1 cosine matching in the Qwen3-Embedding-8B space (paper, Appendix A.1).
3. [qwen_embeddings.py](qwen_embeddings.py): for every dataset, computes Qwen3-Embedding-8B embeddings and the five translated test sets into `embeddings/`.
4. [stage_embeddings.py](stage_embeddings.py): copies these into `detection/results/<seed>/...`, where the experiment scripts pick them up instead of recomputing them. This step is required for the out-of-domain runs on AutoDAN-Turbo+WG.

### 2. In-domain and cross-lingual experiments

```bash
bash scripts/02_indomain.sh          # seeds 0-4; or e.g. `bash scripts/02_indomain.sh 0`
```

For every seed and dataset this runs all baselines and the attention-head probes on both LLMs, on the English test set and its five translations ([main_detection.py](main_detection.py)). On AutoDAN-Turbo+WG, which has no training set, only the zero-shot detectors are run.

A single run, e.g. Llama-3.2-1B probes on WildGuardMix with seed 0:

```bash
PYTHONHASHSEED=42 python main_detection.py --dataset wildguardmix --train train --test test --exp_num 0 \
    --model_name meta-llama/Llama-3.2-1B-Instruct --do_probes --do_translation
```

### 3. Hidden-state probe ablation

```bash
bash scripts/03_hidden_probes.sh     # after step 2: reuses its translated test sets
```

### 4. Out-of-domain experiments

```bash
bash scripts/04_out_of_domain.sh     # seed 0, as in the paper
```

This uses the detectors trained in step 2 ([main_detection_out_of_domain.py](main_detection_out_of_domain.py)). It covers every pair of training set and target, except a dataset with itself and VanillaWildJailbreak ↔ WildJailbreak, since one is a subset of the other.




## Licence

Code released under the [Apache License 2.0](LICENSE). Datasets and models are subject to their own licences.
