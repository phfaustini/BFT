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

import detection.probes_hidden as probes
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
    batch_size = 100
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
        targetlang_csv_path_std = targetlang_csv_path.replace('probeshidden_', '')
        print(f"\nCopying from {targetlang_csv_path_std} into {targetlang_csv_path}")
        if os.name == 'nt':  # Windows
            os.system(f'copy {targetlang_csv_path_std} {targetlang_csv_path}')
        else:  # Linux/Unix
            os.system(f'cp {targetlang_csv_path_std} {targetlang_csv_path}')
    return targetlang_csv_path


if __name__ == "__main__":
    set_deterministic()
    parser = argparse.ArgumentParser(description="Run model inference on a dataset.")
    parser.add_argument("--dataset", type=str, required=False, default="BeaverTails", help="The dataset")
    parser.add_argument("--train", type=str, required=False, default="30k_train", help="split_name_train")
    parser.add_argument("--test", type=str, required=False, default="30k_test", help="split_name_test")
    parser.add_argument("--exp_num", type=int, required=False, default=0, help="Experiment number")
    parser.add_argument("--model_name", type=str, required=False, default="Qwen/Qwen3-8B", help="model_name for probes")
    parser.add_argument("--do_probes", action='store_true', help="Whether to run Probes baseline.")
    parser.add_argument("--do_translation", action='store_true', help="Whether to translate the test set and run the baselines on it. If True, the translated test set will be saved in the output folder, so that it doesn't need to be translated again in future runs.")
    args = parser.parse_args()
    pprint(vars(args))
    split_name_train=args.train
    split_name_test=args.test
    output_folder = os.path.join("detection/results", f"probeshidden_{str(args.exp_num)}", args.dataset)
    os.makedirs(output_folder, exist_ok=True)

    df_test = md.get_dataset(args.dataset, split_name_test)
    df_train = md.get_dataset(args.dataset, split_name_train)
    df_test.to_csv(os.path.join(output_folder, f"df_{split_name_test}_english.csv"), index=False)
    df_train.to_csv(os.path.join(output_folder, f"df_{split_name_train}_english.csv"), index=False)
    set_deterministic(seed=args.exp_num)
    if args.do_translation:
        tlg_dftlg = []
        for target_language in ["spanish", "portuguese", "chinese", "macedonian", "somali"]:
            print(f"\nTranslating test set to {target_language} and running baselines...")
            tlg_csv_path = translate_and_return_path(output_folder, split_name_test, df_test, target_language)
            print(f"Translated test set saved at {tlg_csv_path}\n")
            df_test_tlg = pd.read_csv(tlg_csv_path)
            tlg_dftlg.append((target_language, df_test_tlg))
    else:
        tlg_dftlg = None

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
