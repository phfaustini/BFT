"""
This script simply saves Qwen embeddings created with SentenceTransformers
to the current directory.

Usage: python qwen_embeddings.py
"""


import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import time
import random
from pprint import pprint

import numpy as np
import pandas as pd
import torch

import detection.models_datasets as md
import detection.attack_baselines as attack_baselines
from detection.detection_baselines import convert_text_to_embeddings


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_deterministic():
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


set_deterministic()
parser = argparse.ArgumentParser(description="Run model inference on a dataset.")
parser.add_argument("--dataset", type=str, required=False, default="BeaverTails", help="The dataset")
parser.add_argument("--train", type=str, required=False, default="30k_train", help="split_name_train")
parser.add_argument("--test", type=str, required=False, default="30k_test", help="split_name_test")
parser.add_argument("--model_name", type=str, required=False, default="Qwen/Qwen3-Embedding-8B", help="model_name")
args = parser.parse_args()


if __name__ == "__main__":
    pprint(vars(args))
    os.makedirs("embeddings", exist_ok=True)
    dataset_name = args.dataset
    split_name_train=args.train
    split_name_test=args.test
    model_name = args.model_name
    batch_size = 256

    
    df_test = md.get_dataset(dataset_name, split_name_test)
    if not os.path.exists(os.path.join("embeddings", f"df_{dataset_name}_{split_name_test}_eng.csv")):
        df_test.to_csv(os.path.join("embeddings", f"df_{dataset_name}_{split_name_test}_eng.csv"), index=False)
    start = time.time()
    set_deterministic()
    if not os.path.exists(f"embeddings/embeddings_{dataset_name}_test_eng.npy"):
        embeddings = convert_text_to_embeddings(df_test['query'].to_list(), model_name)
        print(f"Time taken to compute test embeddings: {time.time() - start:.2f} seconds")
        with open(f"embeddings/embeddings_{dataset_name}_test_eng.npy", 'wb') as f:
            np.save(f, embeddings.cpu().float().numpy())

    df_train = md.get_dataset(dataset_name, split_name_train)
    df_train.to_csv(os.path.join("embeddings", f"df_{dataset_name}_{split_name_train}_eng.csv"), index=False)
    start = time.time()
    set_deterministic()
    embeddings = convert_text_to_embeddings(df_train['query'].to_list(), model_name)
    print(f"Time taken to compute train embeddings: {time.time() - start:.2f} seconds")
    with open(f"embeddings/embeddings_{dataset_name}_train_eng.npy", 'wb') as f:
        np.save(f, embeddings.cpu().float().numpy())

    for target_language in ["spanish", "portuguese", "chinese", "macedonian", "somali"]:
        targetlang_csv_path = os.path.join("embeddings", f"df_{dataset_name}_{split_name_test}_{target_language.lower()}.csv")
        if not os.path.exists(targetlang_csv_path):
            print(f"Translating {dataset_name} test queries to {target_language}...")
            targetlang_texts = attack_baselines.translate(df_test['query'].to_list(), target_language)
            df_test_targetlang = df_test.copy()
            df_test_targetlang['query_en'] = df_test['query'].to_list()
            df_test_targetlang['query'] = targetlang_texts
            df_test_targetlang.to_csv(targetlang_csv_path, index=False)
        df_test_tlg = pd.read_csv(targetlang_csv_path)
        df_test_tlg['query'] = df_test_tlg['query'].fillna('').astype(str)
        start = time.time()
        set_deterministic()
        embeddings = convert_text_to_embeddings(df_test_tlg['query'].to_list(), model_name)
        print(f"Time taken to compute {target_language} embeddings: {time.time() - start:.2f} seconds")
        with open(f"embeddings/embeddings_{dataset_name}_test_{target_language}.npy", 'wb') as f:
            np.save(f, embeddings.cpu().float().numpy())


    # with open(f"embeddings/embeddings_{dataset_name}_train_eng.npy", 'rb') as f:
    #   embeddings_train_eng = np.load(f)
