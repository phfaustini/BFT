import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import time
import random
from pickle import load
from pprint import pprint

import numpy as np
import pandas as pd
import torch

import detection.probes as probes
import detection.detection_baselines as detection_baselines
import detection.models_datasets as md


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# All set from argparse in __main__.
TRAIN_DATASET = None
TRAIN_SPLIT = None
TEST_DATASET = None
TEST_SPLIT = None
# In-domain splits used to locate precomputed artefacts from the in-domain runs.
TRAIN_INDOMAIN_TEST_SPLIT = None  # test split of the in-domain --train_dataset run (locates pre-trained models).
TEST_INDOMAIN_TRAIN_SPLIT = None  # train split of the in-domain --test_dataset run (locates precomputed test embeddings).

# Pre-trained classifiers live under the in-domain --train_dataset results.
# Pre-computed test embeddings (of --test_dataset test queries) live under the in-domain --test_dataset results.
TRAIN_INDOMAIN_RESULTS = None
TEST_INDOMAIN_RESULTS = None


def set_deterministic(seed: int = 42):
    if 'cuda' in DEVICE.type:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def run_embeddings(output_folder: str,
                   embedding_name: str,
                   df_test: pd.DataFrame):
    start = time.time()
    print(f"\nRunning Sentence Transformers + Logistic Regression with {embedding_name}...")

    save_folder = os.path.join(output_folder,
                               f"test_{TEST_DATASET}_{TEST_SPLIT}/embeddings_lr/train_{TRAIN_DATASET}_{TRAIN_SPLIT}/model_{embedding_name.replace('/', '_')}/original")
    os.makedirs(save_folder, exist_ok=True)
    if os.path.exists(os.path.join(save_folder, 'roc_curve.png')):
        print(f"Results already exist at {save_folder}, skipping.")
        return

    indomain_test_path = os.path.join(TEST_INDOMAIN_RESULTS,
                                      f"embeddings_lr/train_{TEST_INDOMAIN_TRAIN_SPLIT}/model_{embedding_name.replace('/', '_')}/original",
                                      "embeddings_test.npy")
    assert os.path.exists(indomain_test_path), f"Test embeddings not found at {indomain_test_path}"
    print(f"Loading precomputed test embeddings from {indomain_test_path}...")
    with open(indomain_test_path, 'rb') as f:
        test_embeddings = np.load(f)

    model_subpath = f"embeddings_lr/train_{TRAIN_SPLIT}/model_{embedding_name.replace('/', '_')}/original"
    trained_model_path = os.path.join(TRAIN_INDOMAIN_RESULTS, model_subpath, "embeddings_lr_model.pkl")
    assert os.path.exists(trained_model_path), f"Pre-trained model not found at {trained_model_path}"
    with open(trained_model_path, "rb") as f:
        clf_lr = load(f)
    probas = clf_lr.predict_proba(test_embeddings)
    md.compute_metrics_and_save(probas[:, 1].tolist(),
                                df_test['is_malicious'].to_list(),
                                df_test['query'].to_list(),
                                save_folder)
    end = time.time()
    print(f"Total time sentence-embeddings: {end - start:.2f} seconds")


def run_tfidf(output_folder: str,
              df_test: pd.DataFrame):
    torch.cuda.empty_cache()
    start = time.time()
    print("\nRunning TF-IDF + Logistic Regression...")

    trained_model_path = os.path.join(TRAIN_INDOMAIN_RESULTS,
                                      f"tfidf_lr/train_{TRAIN_SPLIT}/model_logistic_regression/original",
                                      "tfidf_lr_model.pkl")
    assert os.path.exists(trained_model_path), f"Pre-trained TF-IDF model not found at {trained_model_path}"
    with open(trained_model_path, "rb") as f:
        pipeline_lr = load(f)

    save_folder = os.path.join(output_folder,
                               f"test_{TEST_DATASET}_{TEST_SPLIT}/tfidf_lr/train_{TRAIN_DATASET}_{TRAIN_SPLIT}/model_logistic_regression/original")
    os.makedirs(save_folder, exist_ok=True)
    if os.path.exists(os.path.join(save_folder, 'roc_curve.png')):
        print(f"Results already exist at {save_folder}, skipping.")
        return

    probas = pipeline_lr.predict_proba(df_test['query'].to_list())
    md.compute_metrics_and_save(probas[:, 1].tolist(),
                                df_test['is_malicious'].to_list(),
                                df_test['query'].to_list(),
                                save_folder)
    end = time.time()
    print(f"Total time TF-IDF: {end - start:.2f} seconds")


def run_probes(output_folder: str,
               model_name: str,
               df_test: pd.DataFrame,
               top_k: int = 1,
               clf_name: str = 'LogisticRegression'):
    torch.cuda.empty_cache()
    start = time.time()
    print(f"\nRunning Probes detection (top_k={top_k}, clf={clf_name})...")

    probes_folder = os.path.join(TRAIN_INDOMAIN_RESULTS,
                                 f"probes_{clf_name}/train_{TRAIN_SPLIT}/model_{model_name.replace('/', '_')}/original",
                                 "probes")
    assert os.path.exists(probes_folder), f"Pre-trained probes not found at {probes_folder}"

    save_folder = os.path.join(output_folder,
                               f"test_{TEST_DATASET}_{TEST_SPLIT}/probes_{clf_name}/train_{TRAIN_DATASET}_{TRAIN_SPLIT}/model_{model_name.replace('/', '_')}/original")
    top_k_folder = os.path.join(save_folder, f"top_k_{top_k}")
    os.makedirs(top_k_folder, exist_ok=True)
    if os.path.exists(os.path.join(top_k_folder, 'roc_curve.png')):
        print(f"Results already exist at {top_k_folder}, skipping.")
        return

    probes.inference_probes_detection(probes_folder,
                                      top_k_folder,
                                      model_name,
                                      df_test,
                                      pos_label=1,
                                      batch_size=100,
                                      top_k=top_k)
    end = time.time()
    print(f"Total time Probes: {end - start:.2f} seconds")


if __name__ == "__main__":
    set_deterministic()
    parser = argparse.ArgumentParser(description="Run out-of-domain detection: train on one dataset/split, test on another.")
    parser.add_argument("--exp_num", type=int, required=False, default=0, help="Experiment number")
    parser.add_argument("--model_name", type=str, required=False, default="Qwen/Qwen3-8B", help="model_name for probes")
    parser.add_argument("--train_dataset", type=str, default="BeaverTails", help="Dataset whose in-domain run holds the pre-trained classifiers.")
    parser.add_argument("--train_split", type=str, default="30k_train", help="Train split used to fit the pre-trained classifiers.")
    parser.add_argument("--train_indomain_test_split", type=str, default="30k_test", help="Test split of the in-domain --train_dataset run (locates pre-trained model files).")
    parser.add_argument("--test_dataset", type=str, default="TrustAIRLab", help="Out-of-domain dataset to evaluate on.")
    parser.add_argument("--test_split", type=str, default="test", help="Test split of --test_dataset.")
    parser.add_argument("--test_indomain_train_split", type=str, default="train", help="Train split of the in-domain --test_dataset run (locates precomputed test embeddings to skip recomputation).")
    parser.add_argument("--do_tfidf", action='store_true', help="Whether to run TF-IDF baseline.")
    parser.add_argument("--do_embeddings", action='store_true', help="Whether to run Embeddings baseline.")
    parser.add_argument("--do_probes", action='store_true', help="Whether to run Probes baseline.")
    args = parser.parse_args()
    pprint(vars(args))

    TRAIN_DATASET = args.train_dataset
    TRAIN_SPLIT = args.train_split
    TEST_DATASET = args.test_dataset
    TEST_SPLIT = args.test_split
    TRAIN_INDOMAIN_TEST_SPLIT = args.train_indomain_test_split
    TEST_INDOMAIN_TRAIN_SPLIT = args.test_indomain_train_split

    output_folder = os.path.join("detection/results", f"ood_{str(args.exp_num)}", "out_of_domain")
    os.makedirs(output_folder, exist_ok=True)

    TRAIN_INDOMAIN_RESULTS = os.path.join("detection/results", str(args.exp_num), TRAIN_DATASET, f"test_{TRAIN_INDOMAIN_TEST_SPLIT}")
    TEST_INDOMAIN_RESULTS = os.path.join("detection/results", str(args.exp_num), TEST_DATASET, f"test_{TEST_SPLIT}")

    df_test = md.get_dataset(TEST_DATASET, TEST_SPLIT)
    df_test.to_csv(os.path.join(output_folder, f"df_{TEST_DATASET}_{TEST_SPLIT}_english.csv"), index=False)

    print(f"\n{'=' * 60}")
    print(f"Out-of-domain detection")
    print(f"  Train: {TRAIN_DATASET} / {TRAIN_SPLIT} (pre-trained models)")
    print(f"  Test:  {TEST_DATASET} / {TEST_SPLIT} ({len(df_test)} samples)")
    print(f"{'=' * 60}\n")

    set_deterministic(seed=args.exp_num)

    if args.do_tfidf:
        set_deterministic(seed=args.exp_num)
        run_tfidf(output_folder, df_test)

    if args.do_embeddings:
        for embedding_name in ["Qwen/Qwen3-Embedding-8B",]:
                               #"sentence-transformers/all-mpnet-base-v2",]:
            set_deterministic(seed=args.exp_num)
            run_embeddings(output_folder, embedding_name, df_test)

    if args.do_probes:
        for top_k in [1, 3, 5, 10, 15, 20, 30, 50]:
            for clf_name in ['LogisticRegression']:
                set_deterministic(seed=args.exp_num)
                run_probes(output_folder, args.model_name, df_test,
                           top_k=top_k, clf_name=clf_name)
