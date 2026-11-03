"""Copy pre-computed Qwen3-Embedding-8B embeddings and translated test sets
(written to ./embeddings by qwen_embeddings.py) into the results layout that
main_detection.py and main_detection_out_of_domain.py read as a cache:

    detection/results/<n>/<dataset>/df_<test_split>_<lang>.csv
    detection/results/<n>/<dataset>/test_<test_split>/embeddings_lr/train_<train_split>/
        model_Qwen_Qwen3-Embedding-8B/{original,translated2<lang>}/embeddings_{train,test}.npy

For the trainable datasets this only saves compute (the entrypoints compute
missing files themselves). For test-only datasets
(greedy_autodanturbo_wildguardmix) it is required: the out-of-domain runner
expects their test embeddings at this location.

Existing files are never overwritten.

    python stage_embeddings.py                    # all paper datasets, seeds 0-4
    python stage_embeddings.py --exp_nums 0 --datasets wildguardmix
"""
import argparse
import os
import shutil


EMBEDDING_DIR = "embeddings"
RESULTS_DIR = os.path.join("detection", "results")
MODEL_DIR = "model_Qwen_Qwen3-Embedding-8B"
LANGUAGES = ["spanish", "portuguese", "chinese", "somali", "macedonian"]

# dataset -> (train split, test split, has a training set)
DATASETS = {
    "wildguardmix": ("train", "test", True),
    "vanillawildjailbreak": ("train", "test", True),
    "TrustAIRLab": ("train", "test", True),
    "allenai_wildjailbreak": ("train", "test", True),
    "greedy_renellm_wildguardmix": ("train", "test", True),
    "greedy_autodanturbo_wildguardmix": ("train", "test", False),
}


def copy_if_missing(src: str, dst: str) -> None:
    if not os.path.exists(src):
        print(f"  [missing] {src}")
        return
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"  {src} -> {dst}")


def stage(dataset: str, exp_num: int) -> None:
    train_split, test_split, has_train = DATASETS[dataset]
    ds_dir = os.path.join(RESULTS_DIR, str(exp_num), dataset)
    emb_dir = os.path.join(ds_dir, f"test_{test_split}", "embeddings_lr", f"train_{train_split}", MODEL_DIR)
    print(f"[{exp_num}] {dataset}")

    copy_if_missing(os.path.join(EMBEDDING_DIR, f"embeddings_{dataset}_test_eng.npy"),
                    os.path.join(emb_dir, "original", "embeddings_test.npy"))
    if has_train:
        copy_if_missing(os.path.join(EMBEDDING_DIR, f"embeddings_{dataset}_train_eng.npy"),
                        os.path.join(emb_dir, "original", "embeddings_train.npy"))
    for lang in LANGUAGES:
        copy_if_missing(os.path.join(EMBEDDING_DIR, f"embeddings_{dataset}_test_{lang}.npy"),
                        os.path.join(emb_dir, f"translated2{lang}", "embeddings_test.npy"))
        copy_if_missing(os.path.join(EMBEDDING_DIR, f"df_{dataset}_{test_split}_{lang}.csv"),
                        os.path.join(ds_dir, f"df_{test_split}_{lang}.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp_nums", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    args = parser.parse_args()
    for n in args.exp_nums:
        for d in args.datasets:
            stage(d, n)
