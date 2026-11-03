import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import time
import random
from pickle import load
from pprint import pprint
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

import detection.probes as probes
import detection.detection_baselines as detection_baselines
import detection.models_datasets as md
import detection.attack_baselines as attack_baselines


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_deterministic(seed: int=42):
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
                   split_name_train: str,
                   split_name_test: str,
                   df_train: pd.DataFrame,
                   df_test: pd.DataFrame,
                   tlg_dftlg: List[Tuple[str, pd.DataFrame]]=None,
                   random_state: int=42):
    start = time.time()
    print("\nRunning Sentence Transformers + Logistic Regression...")
    save_folder = os.path.join(output_folder, f"test_{split_name_test}/embeddings_lr/train_{split_name_train}/model_{embedding_name.replace('/', '_')}")
    save_folder_original = os.path.join(save_folder, "original")
    os.makedirs(save_folder_original, exist_ok=True)
    if not os.path.exists(os.path.join(save_folder_original, f'roc_curve.png')):
        clf_lr, df_results_original = detection_baselines.run_embedding_detection(save_folder_original,
                                                                                  df_train,
                                                                                  df_test,
                                                                                  batch_size=250,
                                                                                  model_name=embedding_name,
                                                                                  random_state=random_state)
    else:
        with open(os.path.join(save_folder_original, "embeddings_lr_model.pkl"), "rb") as f:
            clf_lr = load(f)

    if tlg_dftlg is not None:
        for (target_language, df_test_targetlang) in tlg_dftlg:
            targetlang_folder = os.path.join(save_folder, f"translated2{target_language}")
            if not os.path.exists(os.path.join(targetlang_folder, f'roc_curve.png')):
                os.makedirs(targetlang_folder, exist_ok=True)
                if os.path.exists(embeddings_test_path := os.path.join(targetlang_folder, "embeddings_test.npy")):
                    print("Loading precomputed test embeddings...")
                    with open(embeddings_test_path, 'rb') as f:
                        test_embeddings = np.load(f)
                else:
                    print("Computing test embeddings...")
                    test_embeddings = detection_baselines.convert_text_to_embeddings(df_test_targetlang['query'].fillna("").to_list(), model_name=embedding_name)
                    test_embeddings = test_embeddings.cpu().float().numpy()
                    with open(embeddings_test_path, 'wb') as f:
                        np.save(f, test_embeddings)
                probas = clf_lr.predict_proba(test_embeddings)
                md.compute_metrics_and_save(probas[:, 1].tolist(),
                                            df_test_targetlang['is_malicious'].to_list(),
                                            df_test_targetlang['query'].fillna("").to_list(),
                                            targetlang_folder)
    end = time.time()
    print(f"Total time sentence-embeddings: {end - start:.2f} seconds")


def run_llama_guard(output_folder: str,
                    split_name_test: str,
                    model_name: str,
                    df_test: pd.DataFrame,
                    tlg_dftlg: List[Tuple[str, pd.DataFrame]]=None,):
    torch.cuda.empty_cache()
    start = time.time()
    print(f"\nRunning Llama Guard detection with {model_name}...")
    save_folder = os.path.join(output_folder, f"test_{split_name_test}/llamaguard/train_none/model_{model_name.replace('/', '_')}")
    save_folder_original = os.path.join(save_folder, "original")
    os.makedirs(save_folder_original, exist_ok=True)
    if not os.path.exists(os.path.join(save_folder_original, f'roc_curve.png')):
        detection_baselines.llama_guard_detection(model_id=model_name,
                                                  save_folder=save_folder_original,
                                                  df_test=df_test, batch_size=32)
    if tlg_dftlg is not None:
        for (target_language, df_test_targetlang) in tlg_dftlg:
            if not os.path.exists(targetlang_folder := os.path.join(save_folder, f"translated2{target_language}")):
                os.makedirs(targetlang_folder, exist_ok=True)
                df_test_targetlang['query'] = df_test_targetlang['query'].fillna("").to_list()
                detection_baselines.llama_guard_detection(model_id=model_name,
                                                          save_folder=targetlang_folder,
                                                          df_test=df_test_targetlang,
                                                          batch_size=32)
    end = time.time()
    print(f"Total time Llama Guard: {end - start:.2f} seconds")


def run_fjd(output_folder: str,
            split_name_test: str,
            model_name: str,
            df_test: pd.DataFrame,
            tlg_dftlg: List[Tuple[str, pd.DataFrame]]=None,):
    torch.cuda.empty_cache()
    start = time.time()
    print(f"\nRunning FJD detection with {model_name}...")
    save_folder = os.path.join(output_folder, f"test_{split_name_test}/fjd/train_none/model_{model_name.replace('/', '_')}")
    save_folder_original = os.path.join(save_folder, "original")
    os.makedirs(save_folder_original, exist_ok=True)
    if not os.path.exists(os.path.join(save_folder_original, f'roc_curve.png')):
        detection_baselines.run_fjd(model_name, save_folder_original, df_test, batch_size=128)

    if tlg_dftlg is not None:
        for (target_language, df_test_targetlang) in tlg_dftlg:
            targetlang_folder = os.path.join(save_folder, f"translated2{target_language}")
            if not os.path.exists(os.path.join(targetlang_folder, 'metrics.json')):
                os.makedirs(targetlang_folder, exist_ok=True)
                detection_baselines.run_fjd(model_name, targetlang_folder, df_test_targetlang, batch_size=128)
    end = time.time()
    print(f"Total time FJD: {end - start:.2f} seconds")


def run_tfidf(output_folder: str,
              split_name_train: str,
              split_name_test: str,
              df_train: pd.DataFrame,
              df_test: pd.DataFrame,
              tlg_dftlg: List[Tuple[str, pd.DataFrame]]=None,
              random_state: int=42):
    torch.cuda.empty_cache()
    start = time.time()
    print("\nRunning TF-IDF + Logistic Regression...")
    save_folder = os.path.join(output_folder, f"test_{split_name_test}/tfidf_lr/train_{split_name_train}/model_logistic_regression")
    save_folder_original = os.path.join(save_folder, "original")
    os.makedirs(save_folder_original, exist_ok=True)
    if not os.path.exists(os.path.join(save_folder_original, f'roc_curve.png')):
        pipeline_lr, df_results_original = detection_baselines.run_tfidf_logistic_regression(save_folder_original,
                                                                        df_train,
                                                                        df_test,
                                                                        random_state=random_state)
    else:
        with open(os.path.join(save_folder_original, "tfidf_lr_model.pkl"), "rb") as f:
            pipeline_lr = load(f)

    if tlg_dftlg is not None:
        for (target_language, df_test_targetlang) in tlg_dftlg:
            if not os.path.exists(targetlang_folder := os.path.join(save_folder, f"translated2{target_language}")):
                os.makedirs(targetlang_folder, exist_ok=True)
                probas = pipeline_lr.predict_proba(df_test_targetlang['query'].fillna("").to_list())
                md.compute_metrics_and_save(probas[:, 1].tolist(),
                                            df_test_targetlang['is_malicious'].to_list(),
                                            df_test_targetlang['query'].fillna("").to_list(),
                                            targetlang_folder)
    end = time.time()
    print(f"Total time TF-IDF: {end - start:.2f} seconds")


def run_ppl(output_folder: str,
            split_name_test: str,
            df_test: pd.DataFrame,
            tlg_dftlg: List[Tuple[str, pd.DataFrame]]=None,):
    torch.cuda.empty_cache()
    start = time.time()
    print("\nRunning Perplexity detection...")
    save_folder = os.path.join(output_folder, f"test_{split_name_test}/perplexity/train_none/model_gptneo")
    save_folder_original = os.path.join(save_folder, "original")
    os.makedirs(save_folder_original, exist_ok=True)
    if not os.path.exists(os.path.join(save_folder_original, f'roc_curve.png')):
        detection_baselines.run_perplexity(save_folder_original, df_test, DEVICE)
    if tlg_dftlg is not None:
        for (target_language, df_test_targetlang) in tlg_dftlg:
            if not os.path.exists(targetlang_folder := os.path.join(save_folder, f"translated2{target_language}")):
                os.makedirs(targetlang_folder, exist_ok=True)
                detection_baselines.run_perplexity(targetlang_folder, df_test_targetlang)
    end = time.time()
    print(f"Total time Perplexity: {end - start:.2f} seconds")


def run_probes(output_folder: str,
               split_name_train: str,
               split_name_test: str,
               model_name: str,
               df_train: pd.DataFrame,
               df_test: pd.DataFrame,
               tlg_dftlg: List[Tuple[str, pd.DataFrame]]=None,
               top_k: int=1,
               clf_name: str='LogisticRegression',
               random_state: int=42):
    torch.cuda.empty_cache()
    start = time.time()
    print("\nRunning Probes detection...")
    save_folder = os.path.join(output_folder, f"test_{split_name_test}/probes_{clf_name}/train_{split_name_train}/model_{model_name.replace('/', '_')}")
    save_folder_original = os.path.join(save_folder, "original")
    batch_size = 200
    os.makedirs(save_folder_original, exist_ok=True)
    probes_folder = os.path.join(save_folder_original, "probes")
    if not os.path.exists(os.path.join(save_folder_original, f"top_k_{top_k}", f'roc_curve.png')):
        probes.run_probes_detection(model_name, save_folder_original,
                                    df_train, df_test,
                                    batch_size=batch_size, top_k=top_k, clf_name=clf_name,
                                    random_state=random_state)
    if tlg_dftlg is not None:
        for (target_language, df_test_targetlang) in tlg_dftlg:
            if not os.path.exists(os.path.join(save_folder, f"translated2{target_language}", f"top_k_{top_k}", f'roc_curve.png')):
                targetlang_folder = os.path.join(save_folder, f"translated2{target_language}", f"top_k_{top_k}")
                os.makedirs(targetlang_folder, exist_ok=True)
                probes.inference_probes_detection(probes_folder, targetlang_folder, model_name,
                                                  df_test_targetlang, pos_label=1,
                                                  batch_size=batch_size, top_k=top_k)
    end = time.time()
    print(f"Total time Probes: {end - start:.2f} seconds")


def translate_and_return_path(output_folder: str,
                              split_name_test: str,
                              df_test: pd.DataFrame,
                              target_language: str="spanish") -> str:
    targetlang_csv_path = os.path.join(output_folder, f"df_{split_name_test}_{target_language.lower()}.csv")
    if not os.path.exists(targetlang_csv_path):
        print(f"\nTranslating test set to {target_language} into {targetlang_csv_path}")
        targetlang_texts = attack_baselines.translate(df_test['query'].fillna("").to_list(), target_language)
        df_test_targetlang = df_test.copy()
        df_test_targetlang['query_en'] = df_test['query'].to_list()
        df_test_targetlang['query'] = targetlang_texts
        df_test_targetlang.to_csv(targetlang_csv_path, index=False)
    return targetlang_csv_path


if __name__ == "__main__":
    set_deterministic()
    parser = argparse.ArgumentParser(description="Run model inference on a dataset.")
    parser.add_argument("--dataset", type=str, required=False, default="BeaverTails", help="The dataset")
    parser.add_argument("--train", type=str, required=False, default="30k_train", help="split_name_train")
    parser.add_argument("--test", type=str, required=False, default="30k_test", help="split_name_test")
    parser.add_argument("--exp_num", type=int, required=False, default=0, help="Experiment number")
    parser.add_argument("--model_name", type=str, required=False, default="Qwen/Qwen3-8B", help="model_name for probes")
    parser.add_argument("--do_tfidf", action='store_true', help="Whether to run TF-IDF baseline.")
    parser.add_argument("--do_embeddings", action='store_true', help="Whether to run Embeddings baseline.")
    parser.add_argument("--do_ppl", action='store_true', help="Whether to run Perplexity baseline.")
    parser.add_argument("--do_llamaguard", action='store_true', help="Whether to run LlamaGuard baseline.")
    parser.add_argument("--do_fjd", action='store_true', help="Whether to run FJD baseline.")
    parser.add_argument("--do_probes", action='store_true', help="Whether to run Probes baseline.")
    parser.add_argument("--do_translation", action='store_true', help="Whether to translate the test set and run the baselines on it. If True, the translated test set will be saved in the output folder, so that it doesn't need to be translated again in future runs.")
    args = parser.parse_args()
    pprint(vars(args))
    split_name_train=args.train
    split_name_test=args.test
    output_folder = os.path.join("detection/results", str(args.exp_num), args.dataset)
    os.makedirs(output_folder, exist_ok=True)

    df_test = md.get_dataset(args.dataset, split_name_test)
    df_train = md.get_dataset(args.dataset, split_name_train)
    df_test.to_csv(os.path.join(output_folder, f"df_{split_name_test}_english.csv"), index=False)
    df_train.to_csv(os.path.join(output_folder, f"df_{split_name_train}_english.csv"), index=False)
    set_deterministic(seed=args.exp_num)
    if args.do_translation:
        tlg_dftlg = []
        for target_language in ["spanish", "portuguese", "chinese", "somali", "macedonian"]:
            print(f"\nTranslating test set to {target_language} and running baselines...")
            tlg_csv_path = translate_and_return_path(output_folder, split_name_test, df_test, target_language)
            print(f"Translated test set saved at {tlg_csv_path}\n")
            df_test_tlg = pd.read_csv(tlg_csv_path)
            tlg_dftlg.append((target_language, df_test_tlg))
    else:
        tlg_dftlg = None

    if args.do_tfidf:
        set_deterministic(seed=args.exp_num)
        run_tfidf(output_folder,
                  split_name_train,
                  split_name_test,
                  df_train,
                  df_test,
                  tlg_dftlg,
                  random_state=args.exp_num)

    if args.do_ppl:
        set_deterministic(seed=args.exp_num)
        run_ppl(output_folder,
                split_name_test,
                df_test,
                tlg_dftlg)

    if args.do_fjd:
        for fjd_model in ["Qwen/Qwen3-8B", "meta-llama/Llama-3.2-1B-Instruct"]:
            set_deterministic(seed=args.exp_num)
            run_fjd(output_folder,
                    split_name_test,
                    fjd_model,
                    df_test,
                    tlg_dftlg)

    if args.do_llamaguard:
        set_deterministic(seed=args.exp_num)
        for llamaguard_name in ["meta-llama/Llama-Guard-3-8B", "meta-llama/Llama-Prompt-Guard-2-86M", "MickyMike/DecipherGuard"]:
            run_llama_guard(output_folder,
                            split_name_test,
                            llamaguard_name,
                            df_test,
                            tlg_dftlg)

    if args.do_embeddings:
        for embedding_name in ["sentence-transformers/all-mpnet-base-v2",
                               "Qwen/Qwen3-Embedding-8B"]:
            print(f"\nRunning Embeddings + Logistic Regression with {embedding_name}...")
            set_deterministic(seed=args.exp_num)
            run_embeddings(output_folder,
                           embedding_name,
                           split_name_train,
                           split_name_test,
                           df_train, df_test,
                           tlg_dftlg,
                           args.exp_num)

    if args.do_probes:
        for top_k in [1, 3, 5, 10, 15, 20, 30, 50]:
            for clf_name in ['LogisticRegression']:  # ['LogisticRegression', 'SVM', 'RandomForest']:
                print(f"\nRunning Probes with {clf_name} and top_k={top_k}...")
                set_deterministic(seed=args.exp_num)
                run_probes(output_folder,
                           split_name_train,
                           split_name_test,
                           args.model_name,
                           df_train,
                           df_test,
                           tlg_dftlg,
                           top_k=top_k,
                           clf_name=clf_name,
                           random_state=args.exp_num)
