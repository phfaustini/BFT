import os
import json
import requests
from typing import List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from datasets import load_dataset, load_from_disk
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.model_selection import train_test_split


def get_dataset(dataset_name: str, split_name: str='330k_train') -> pd.DataFrame:
    if dataset_name.lower() == "advbench":
        path = "data/AdvBench/harmful_behaviors_custom.csv"
        df = pd.read_csv(path)[['goal', 'target', 'category', 'Original index']]
        df.rename(columns={'goal': 'query'}, inplace=True)
    elif dataset_name.lower() == "fft":
        path = "data/FFT/Toxicity/Toxic_questions.json"
        df = pd.read_json(path)
    elif dataset_name.lower() == "autodanturbo":
        os.makedirs('data/AutoDanTurboLogs', exist_ok=True)
        if not os.path.exists(autodanlog_path:='data/AutoDanTurboLogs/lifelong_attack_log.json'):
            url = 'https://github.com/SaFo-Lab/AutoDAN-Turbo/raw/refs/heads/main/logs_r/lifelong_attack_log.json'
            r = requests.get(url)
            open(autodanlog_path , 'wb').write(r.content)
        with open(autodanlog_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        df = {"original": [], "query": [], "score": [], "target": []}
        for row in json_data:
            if row['stage'] == 'lifelong' and row['score'] >= 8.5:
                df['original'].append(row['request'])
                df['query'].append(row['prompt'])
                df['score'].append(row['score'])
                df['target'].append(row['response'])
        df = pd.DataFrame(df)
        df['is_malicious'] = 1
        df['split'] = 'test'
    elif dataset_name.lower() == "autodanturbo_beavertails":
        if os.path.exists('data/combined_autodanturbo_beavertails/combined.csv'):
            df = pd.read_csv('data/combined_autodanturbo_beavertails/combined.csv')
        else:
            os.makedirs('data/combined_autodanturbo_beavertails', exist_ok=True)
            os.makedirs('data/AutoDanTurboLogs', exist_ok=True)
            if not os.path.exists(autodanlog_path:='data/AutoDanTurboLogs/lifelong_attack_log.json'):
                url = 'https://github.com/SaFo-Lab/AutoDAN-Turbo/raw/refs/heads/main/logs_r/lifelong_attack_log.json'
                r = requests.get(url)
                open(autodanlog_path , 'wb').write(r.content)
            with open(autodanlog_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            autodan_df = {"original": [], "query": [], "score": [], "target": []}
            for row in json_data:
                if row['stage'] == 'lifelong' and row['score'] >= 8.5:
                    autodan_df['original'].append(row['request'])
                    autodan_df['query'].append(row['prompt'])
                    autodan_df['score'].append(row['score'])
                    autodan_df['target'].append(row['response'])
            autodan_df = pd.DataFrame(autodan_df)
            autodan_df['is_malicious'] = 1
            autodan_df['split'] = 'test'
            bt_test_df = get_dataset("beavertails", "30k_test")
            bt_test_df['split'] = 'test'
            bt_test_benign_df = bt_test_df[bt_test_df['is_malicious']==0].copy()
            bt_test_benign_df = bt_test_benign_df.sample(n=autodan_df.shape[0], random_state=42)
            df = pd.concat([bt_test_benign_df, autodan_df], ignore_index=True)
            df.to_csv('data/combined_autodanturbo_beavertails/combined.csv', index=False)
    elif dataset_name.lower() == "autodanturbo_vanillawildjailbreak":
        if os.path.exists('data/combined_autodanturbo_vanillawildjailbreak/combined.csv'):
            df = pd.read_csv('data/combined_autodanturbo_vanillawildjailbreak/combined.csv')
        else:
            os.makedirs('data/combined_autodanturbo_vanillawildjailbreak', exist_ok=True)
            os.makedirs('data/AutoDanTurboLogs', exist_ok=True)
            if not os.path.exists(autodanlog_path:='data/AutoDanTurboLogs/lifelong_attack_log.json'):
                url = 'https://github.com/SaFo-Lab/AutoDAN-Turbo/raw/refs/heads/main/logs_r/lifelong_attack_log.json'
                r = requests.get(url)
                open(autodanlog_path , 'wb').write(r.content)
            with open(autodanlog_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            autodan_df = {"original": [], "query": [], "score": [], "target": []}
            for row in json_data:
                if row['stage'] == 'lifelong' and row['score'] >= 8.5:
                    autodan_df['original'].append(row['request'])
                    autodan_df['query'].append(row['prompt'])
                    autodan_df['score'].append(row['score'])
                    autodan_df['target'].append(row['response'])
            autodan_df = pd.DataFrame(autodan_df)
            autodan_df['is_malicious'] = 1
            autodan_df['split'] = 'test'
            vw_test_df = get_dataset("vanillawildjailbreak", "test")
            vw_test_df['split'] = 'test'
            vw_test_benign_df = vw_test_df[vw_test_df['is_malicious']==0].copy()
            vw_test_benign_df = vw_test_benign_df.sample(n=autodan_df.shape[0], random_state=42)
            df = pd.concat([vw_test_benign_df, autodan_df], ignore_index=True)
            df.to_csv('data/combined_autodanturbo_vanillawildjailbreak/combined.csv', index=False)
    elif dataset_name.lower() == "greedy_autodanturbo_wildguardmix":
        if os.path.exists('data/combined_greedy_autodanturbo_wildguardmix/combined.csv'):
            df = pd.read_csv('data/combined_greedy_autodanturbo_wildguardmix/combined.csv')
        else:
            os.makedirs('data/combined_greedy_autodanturbo_wildguardmix', exist_ok=True)
            os.makedirs('data/AutoDanTurboLogs', exist_ok=True)
            if not os.path.exists(autodanlog_path:='data/AutoDanTurboLogs/lifelong_attack_log.json'):
                url = 'https://github.com/SaFo-Lab/AutoDAN-Turbo/raw/refs/heads/main/logs_r/lifelong_attack_log.json'
                r = requests.get(url)
                open(autodanlog_path , 'wb').write(r.content)
            with open(autodanlog_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            autodan_df = {"original": [], "query": [], "score": [], "target": []}
            for row in json_data:
                if row['stage'] == 'lifelong' and row['score'] >= 8.5:
                    autodan_df['original'].append(row['request'])
                    autodan_df['query'].append(row['prompt'])
                    autodan_df['score'].append(row['score'])
                    autodan_df['target'].append(row['response'])
            autodan_df = pd.DataFrame(autodan_df)
            autodan_df['is_malicious'] = 1
            autodan_df['split'] = 'test'
            wg_test_df = get_dataset("wildguardmix", "test").reset_index(drop=True)
            wg_test_df['split'] = 'test'
            # Pick wg benign rows closest to the autodan attacks by cosine similarity
            # (greedy 1:1: each autodan row claims its most-similar still-unused wg benign row).
            with open(f"embeddings/embeddings_wildguardmix_test_eng.npy", 'rb') as f:
                emb_wg = np.load(f)
            from detection.detection_baselines import convert_text_to_embeddings
            emb_at = convert_text_to_embeddings(autodan_df['query'].tolist(), 'Qwen/Qwen3-Embedding-8B', batch_size=16)
            emb_at = emb_at.detach().to('cpu').float().numpy()
            wg_b_pos = np.where((wg_test_df['is_malicious']==0).to_numpy())[0]
            emb_wg_b = emb_wg[wg_b_pos]
            wg_test_benign_df = wg_test_df.iloc[wg_b_pos].reset_index(drop=True)
            at_norm = emb_at / np.linalg.norm(emb_at, axis=1, keepdims=True)
            wgb_norm = emb_wg_b / np.linalg.norm(emb_wg_b, axis=1, keepdims=True)
            cos_sim = at_norm @ wgb_norm.T  # (n_at, n_wg_b)
            n_at, n_wg_b = cos_sim.shape
            assert n_at <= n_wg_b, "Not enough wg benign rows to do 1:1 matching."
            picked = np.empty(n_at, dtype=np.int64)
            used = np.zeros(n_wg_b, dtype=bool)
            for i in range(n_at):
                sims = cos_sim[i].copy()
                sims[used] = -np.inf
                j = int(np.argmax(sims))
                picked[i] = j
                used[j] = True
            wg_test_benign_df = wg_test_benign_df.iloc[picked].reset_index(drop=True)
            # Annotate each paired row with its match: every benign row stores the autodan
            # query it was paired with, every autodan row stores the benign query it picked,
            # and both share the cosine similarity of that pair.
            matched_sims = cos_sim[np.arange(n_at), picked]
            wg_test_benign_df['closest_query'] = autodan_df['query'].values
            wg_test_benign_df['cos_sim'] = matched_sims
            autodan_df['closest_query'] = wg_test_benign_df['query'].values
            autodan_df['cos_sim'] = matched_sims
            df = pd.concat([wg_test_benign_df, autodan_df], ignore_index=True)
            df.to_csv('data/combined_greedy_autodanturbo_wildguardmix/combined.csv', index=False)
            # Save combined embeddings aligned row-for-row with combined.csv
            # ([wg_benign_picked, autodan]).
            os.makedirs('embeddings', exist_ok=True)
            emb_combined = np.concatenate([emb_wg_b[picked], emb_at], axis=0)
            with open('embeddings/embeddings_greedy_autodanturbo_wildguardmix_test_eng.npy', 'wb') as f:
                np.save(f, emb_combined)
    elif dataset_name.lower() == "renellm_beavertails":
        if os.path.exists('data/combined_renellm_beavertails/combined.csv'):
            df = pd.read_csv('data/combined_renellm_beavertails/combined.csv')
        else:
            if not os.path.exists('data/ReNeLLM-Jailbreak/renellm_train.csv'):
                ds = load_from_disk('data/ReNeLLM-Jailbreak')
                df_rene = ds["train"].to_pandas() # Only train available.
                df_rene['is_malicious'] = 1
                df_rene['split'] = 'train'
                os.makedirs('data/ReNeLLM-Jailbreak', exist_ok=True)
                df_rene.to_csv('data/ReNeLLM-Jailbreak/renellm_train.csv', index=False)
            else:
                df_rene = pd.read_csv('data/ReNeLLM-Jailbreak/renellm_train.csv')
                df_rene['split'] = 'train'
            if not os.path.exists('data/BeaverTails'):
                dataset = load_dataset('PKU-Alignment/BeaverTails')
                os.makedirs('data/BeaverTails')
                dataset.save_to_disk("data/BeaverTails")  # dataset = load_from_disk("data/BeaverTails")
            else:
                dataset = load_from_disk("data/BeaverTails")
            df_beaver_train = dataset["330k_train"].to_pandas()
            df_beaver_test = dataset["330k_test"].to_pandas()
            df_beaver_train['is_malicious'] = df_beaver_train['is_safe'].apply(lambda x: 0 if x==True else 1)
            df_beaver_test['is_malicious'] = df_beaver_test['is_safe'].apply(lambda x: 0 if x==True else 1)
            df_beaver_train = df_beaver_train[df_beaver_train['is_malicious']==0].copy()
            df_beaver_test = df_beaver_test[df_beaver_test['is_malicious']==0].copy()
            del df_beaver_train['is_safe']
            del df_beaver_test['is_safe']
            categories_train, categories_test = [], []
            for row_dict in df_beaver_train['category'].to_list():  # {'animal_abuse': True, 'child_abuse': False}
                row_cats_train = [k for k, v in row_dict.items() if v==True]  # ['animal_abuse']
                categories_train.append(sorted(row_cats_train))
            for row_dict in df_beaver_test['category'].to_list():  # {'animal_abuse': True, 'child_abuse': False}
                row_cats_test = [k for k, v in row_dict.items() if v==True]  # ['animal_abuse']
                categories_test.append(sorted(row_cats_test))
            df_beaver_train['category'] = categories_train
            df_beaver_test['category'] = categories_test
            df_beaver_train.rename(columns={'prompt': 'query', 'response': 'target'}, inplace=True)
            df_beaver_test.rename(columns={'prompt': 'query', 'response': 'target'}, inplace=True)

            df_rene.rename(columns={'nested_prompt': 'query', 'claude2_output': 'target'}, inplace=True)
            df_rene['is_malicious'] = 1
            df_beaver_train.drop_duplicates(subset=['query'], inplace=True)
            df_beaver_test.drop_duplicates(subset=['query'], inplace=True)
            df_rene.drop_duplicates(subset=['query'], inplace=True)
            n_test_each = 1140 # df_beaver_test.shape[0]  # 1,140 benign BeaverTails 30k set.
            n_train_each = 4555  # df_beaver_train.shape[0]  # 4,555 benign BeaverTails 30k set.

            #df_rene_train, df_beaver_train, df_rene_test, df_beaver_test = _compute_similar(df_rene, df_beaver_train, df_beaver_test, n_train_each, n_test_each)

            # Group-aware split on original_harm_behavior: every nested_prompt
            # derived from the same source sentence must land in the same split.
            # Sizes are approximate (groups are kept intact) — they slightly
            # overshoot the n_*_each targets by at most one group.
            group_sizes = df_rene.groupby('original_harm_behavior').size()
            shuffled_groups = np.random.default_rng(42).permutation(group_sizes.index.values)
            test_groups, n_test = [], 0
            for g in shuffled_groups:
                if n_test >= n_test_each:
                    break
                test_groups.append(g)
                n_test += group_sizes[g]
            test_groups = set(test_groups)
            train_groups, n_train = [], 0
            for g in shuffled_groups:
                if g in test_groups:
                    continue
                if n_train >= n_train_each:
                    break
                train_groups.append(g)
                n_train += group_sizes[g]
            train_groups = set(train_groups)
            df_rene_test = df_rene[df_rene['original_harm_behavior'].isin(test_groups)].copy()
            df_rene_train = df_rene[df_rene['original_harm_behavior'].isin(train_groups)].copy()
            df_rene_train['split'] = 'train'
            df_rene_test['split'] = 'test'

            df = pd.concat([df_rene_train, df_beaver_train, df_rene_test, df_beaver_test], ignore_index=True)
            os.makedirs('data/combined_renellm_beaver', exist_ok=True)
            df.to_csv('data/combined_renellm_beaver/combined.csv', index=False)
    elif dataset_name.lower() == "renellm_wildguardmix":
        if os.path.exists('data/combined_renellm_wildguardmix/combined.csv'):
            df = pd.read_csv('data/combined_renellm_wildguardmix/combined.csv')
        else:
            if not os.path.exists('data/ReNeLLM-Jailbreak/renellm_train.csv'):
                ds = load_from_disk('data/ReNeLLM-Jailbreak')
                df_rene = ds["train"].to_pandas() # Only train available.
                df_rene['is_malicious'] = 1
                df_rene['split'] = 'train'
                os.makedirs('data/ReNeLLM-Jailbreak', exist_ok=True)
                df_rene.to_csv('data/ReNeLLM-Jailbreak/renellm_train.csv', index=False)
            else:
                df_rene = pd.read_csv('data/ReNeLLM-Jailbreak/renellm_train.csv')
                df_rene['split'] = 'train'
            df_wildguardmix_train = get_dataset('wildguardmix', 'train')
            df_benign_train = df_wildguardmix_train[df_wildguardmix_train['is_malicious']==0].copy()
            df_wildguardmix_test = get_dataset('wildguardmix', 'test')
            df_test_benign = df_wildguardmix_test[df_wildguardmix_test['is_malicious']==0].copy()

            df_rene.rename(columns={'nested_prompt': 'query', 'claude2_output': 'target'}, inplace=True)
            df_rene['is_malicious'] = 1
            df_rene.drop_duplicates(subset=['query'], inplace=True)
            n_test_each = df_test_benign.shape[0] # df_beaver_test.shape[0]  # 971 benign
            n_train_each = df_benign_train.shape[0]  # 23084 benign

            # Group-aware split on original_harm_behavior: every nested_prompt
            # derived from the same source sentence must land in the same split.
            # Sizes are approximate (groups are kept intact) — they slightly
            # overshoot the n_*_each targets by at most one group.
            group_sizes = df_rene.groupby('original_harm_behavior').size()
            shuffled_groups = np.random.default_rng(42).permutation(group_sizes.index.values)
            test_groups, n_test = [], 0
            for g in shuffled_groups:
                if n_test >= n_test_each:
                    break
                test_groups.append(g)
                n_test += group_sizes[g]
            test_groups = set(test_groups)
            train_groups, n_train = [], 0
            for g in shuffled_groups:
                if g in test_groups:
                    continue
                if n_train >= n_train_each:
                    break
                train_groups.append(g)
                n_train += group_sizes[g]
            train_groups = set(train_groups)
            df_rene_test = df_rene[df_rene['original_harm_behavior'].isin(test_groups)].copy()
            df_rene_train = df_rene[df_rene['original_harm_behavior'].isin(train_groups)].copy()
            df_rene_train['split'] = 'train'
            df_rene_test['split'] = 'test'

            df = pd.concat([df_rene_train, df_benign_train, df_rene_test, df_test_benign], ignore_index=True)
            os.makedirs('data/combined_renellm_wildguardmix', exist_ok=True)
            df.to_csv('data/combined_renellm_wildguardmix/combined.csv', index=False)
        if 'train' in split_name:
            df = df[df["split"]=='train']
        else:  # if 'test' in split_name:
            df = df[df["split"]=='test']
    elif dataset_name.lower() == "greedy_renellm_wildguardmix":
        df = pd.read_csv('data/combined_greedy_renellm_wildguardmix/combined.csv')
        if 'train' in split_name:
            df = df[df["split"]=='train']
        else:  # if 'test' in split_name:
            df = df[df["split"]=='test']
    elif dataset_name.lower() == "renellm":
        if not os.path.exists('data/ReNeLLM-Jailbreak/renellm_train.csv'):
            ds = load_dataset('Deep1994/ReNeLLM-Jailbreak')
            df_rene = ds["train"].to_pandas() # Only train available.
            os.makedirs('data/ReNeLLM-Jailbreak', exist_ok=True)
            df_rene.to_csv('data/ReNeLLM-Jailbreak/renellm_train.csv', index=False)
        else:
            df = pd.read_csv('data/ReNeLLM-Jailbreak/renellm_train.csv')
        df.rename(columns={'nested_prompt': 'query', 'claude2_output': 'target'}, inplace=True)
        df['is_malicious'] = 1
        df['split'] = 'train'
    elif dataset_name.lower() == "beavertails":
        if not os.path.exists('data/BeaverTails'):
            dataset = load_dataset('PKU-Alignment/BeaverTails')
            os.makedirs('data/BeaverTails')
            dataset.save_to_disk("data/BeaverTails")  # dataset = load_from_disk("data/BeaverTails")
        else:
            dataset = load_from_disk("data/BeaverTails")
        df = dataset[split_name].to_pandas()
        df['is_malicious'] = df['is_safe'].apply(lambda x: 0 if x==True else 1)
        del df['is_safe']
        categories = []
        for row_dict in df['category'].to_list():  # {'animal_abuse': True, 'child_abuse': False}
            row_cats = [k for k, v in row_dict.items() if v==True]  # ['animal_abuse']
            categories.append(sorted(row_cats))
        df['category'] = categories
        df.rename(columns={'prompt': 'query', 'response': 'target'}, inplace=True)
    elif "vanillawildjailbreak" in dataset_name.lower():
        if not os.path.exists('data/allenai_wildjailbreak/allenai_wildjailbreak_vanilla.csv'):
            if not os.path.exists('data/allenai_wildjailbreak'):
                dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
                os.makedirs('data/allenai_wildjailbreak', exist_ok=True)
                dataset.save_to_disk("data/allenai_wildjailbreak")
            else:
                dataset = load_from_disk("data/allenai_wildjailbreak")
            df = dataset['train'].to_pandas()  # ['vanilla', 'adversarial', 'completion', 'data_type']
            df = df[(df['data_type']=='vanilla_harmful')|(df['data_type']=='vanilla_benign')].copy()
            df["is_malicious"] = df['data_type'].apply(lambda x: 0 if x=="vanilla_benign" else 1)
            df.drop_duplicates(subset=['vanilla'], inplace=True, keep='first')
            df.rename(columns={'vanilla': 'query', 'completion': 'target'}, inplace=True)
            df_train, df_test = train_test_split(df, test_size=0.3, random_state=0, stratify=df[['is_malicious']])
            
            df_train['split'] = 'train'
            df_test['split'] = 'test'
            df = pd.concat([df_train, df_test], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
            del df['adversarial']
            df.to_csv(os.path.join('data/allenai_wildjailbreak', 'allenai_wildjailbreak_vanilla.csv'), index=False)
        else:
            df = pd.read_csv(os.path.join('data/allenai_wildjailbreak', 'allenai_wildjailbreak_vanilla.csv'))
        if 'train' in split_name:
            df = df[df["split"]=='train']
        else:  # if 'test' in split_name:
            df = df[df["split"]=='test']
    elif "wildjailbreak" in dataset_name.lower():
        if not os.path.exists('data/allenai_wildjailbreak/allenai_wildjailbreak_subset.csv'):
            if not os.path.exists('data/allenai_wildjailbreak'):
                dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
                os.makedirs('data/allenai_wildjailbreak', exist_ok=True)
                dataset.save_to_disk("data/allenai_wildjailbreak")
            else:
                dataset = load_from_disk("data/allenai_wildjailbreak")
            df = dataset['train'].to_pandas()  # ['vanilla', 'adversarial', 'completion', 'data_type']
            df["is_malicious"] = df['data_type'].apply(lambda x: 0 if x=="vanilla_benign" or x=="adversarial_benign" else 1)
            df_train, df_test = train_test_split(df, test_size=0.3, random_state=0, stratify=df[['is_malicious']])
            out_train = {"query": [], "target": [], "is_malicious": [], "data_type": []}
            for i, row in df_train.iterrows():
                out_train["query"].append(row['vanilla'].strip())
                out_train["target"].append(row['completion'])
                out_train["is_malicious"].append(row["is_malicious"])
                out_train["data_type"].append(row["data_type"])
                if row['adversarial'].strip() != "":
                    out_train["query"].append(row['adversarial'].strip())
                    out_train["target"].append(row['completion'])
                    out_train["is_malicious"].append(row["is_malicious"])
                    out_train["data_type"].append(row["data_type"])
            out_test = {"query": [], "target": [], "is_malicious": [], "data_type": []}
            for i, row in df_test.iterrows():
                out_test["query"].append(row['vanilla'].strip())
                out_test["target"].append(row['completion'])
                out_test["is_malicious"].append(row["is_malicious"])
                out_test["data_type"].append(row["data_type"])
                if row['adversarial'].strip() !="":
                    out_test["query"].append(row['adversarial'].strip())
                    out_test["target"].append(row['completion'])
                    out_test["is_malicious"].append(row["is_malicious"])
                    out_test["data_type"].append(row["data_type"])
            df_train = pd.DataFrame(out_train)
            df_test = pd.DataFrame(out_test)
            df_train['split'] = 'train'
            df_test['split'] = 'test'
            df = pd.concat([df_train, df_test], ignore_index=True)
            df.drop_duplicates(subset=['query'], inplace=True, keep='first')
            df_train = df[df['split']=='train'].copy()
            df_test = df[df['split']=='test'].copy()

            df_train_balanced = (
                df_train.groupby('data_type', group_keys=False)
                .apply(lambda x: x.sample(n=7000, random_state=42))
            )
            df_test_balanced = (
                df_test.groupby('data_type', group_keys=False)
                .apply(lambda x: x.sample(n=1000, random_state=42))
            )
            # Get remaining rows using index difference
            df_train_remaining = df_train.loc[df_train.index.difference(df_train_balanced.index)]
            df_test_remaining = df_test.loc[df_test.index.difference(df_test_balanced.index)]

            # Now reset index for the balanced sets
            df_train_balanced = df_train_balanced.reset_index(drop=True)
            df_test_balanced = df_test_balanced.reset_index(drop=True)
            df = pd.concat([df_train_balanced, df_test_balanced], ignore_index=True)
            df.drop_duplicates(subset=['query'], inplace=True, keep='first')
            df.to_csv(os.path.join('data/allenai_wildjailbreak', 'allenai_wildjailbreak_subset.csv'), index=False)

            df_remaining = pd.concat([df_train_remaining, df_test_remaining], ignore_index=True)
            df_remaining.drop_duplicates(subset=['query'], inplace=True, keep='first')
            df_remaining.to_csv(os.path.join('data/allenai_wildjailbreak', 'allenai_wildjailbreak_subsetremaining.csv'), index=False)
        else:
            df = pd.read_csv(os.path.join('data/allenai_wildjailbreak', 'allenai_wildjailbreak_subset.csv'))
        if 'train' in split_name:
            df = df[df["split"]=='train']
        else:  # if 'test' in split_name:
            df = df[df["split"]=='test']
    elif dataset_name.lower() == "wildguardmix":
        if not os.path.exists('data/wildguardmix/wildguardmix.csv'):
            os.makedirs('data/wildguardmix', exist_ok=True)
            ds_train = load_dataset("allenai/wildguardmix", "wildguardtrain")
            ds_test = load_dataset("allenai/wildguardmix", "wildguardtest")
            ds_train.save_to_disk("data/wildguardmix")
            ds_test.save_to_disk("data/wildguardmix")
            df_test = ds_test['test'].to_pandas()
            df_train = ds_train['train'].to_pandas()
            df_train = df_train[['prompt', 'response', 'prompt_harm_label']].copy()
            df_test = df_test[['prompt', 'response', 'prompt_harm_label']].copy()
            df_train.rename(columns={'prompt': 'query', 'response': 'target'}, inplace=True)
            df_train['is_malicious'] = df_train['prompt_harm_label'].apply(lambda x: 1 if x=='harmful' else 0)
            del df_train['prompt_harm_label']
            df_test.rename(columns={'prompt': 'query', 'response': 'target'}, inplace=True)
            df_test['is_malicious'] = df_test['prompt_harm_label'].apply(lambda x: 1 if x=='harmful' else 0)
            del df_test['prompt_harm_label']
            df_train.drop_duplicates(subset=['query'], inplace=True)
            df_train['split'] = 'train'
            df_train = df_train[df_train['query'].str.len()>0]
            df_test.drop_duplicates(subset=['query'], inplace=True)
            df_test['split'] = 'test'
            df = pd.concat([df_train, df_test.dropna()], ignore_index=True)
            df.to_csv('data/wildguardmix/wildguardmix.csv', index=False)
        else:
            df = pd.read_csv('data/wildguardmix/wildguardmix.csv')
        if 'train' in split_name:
            df = df[df["split"]=='train']
        else:  # if 'test' in split_name:
            df = df[df["split"]=='test']
    else:
        if not os.path.exists('data/TrustAIRLab'):
            os.makedirs('data/TrustAIRLab/wild_jailbreak_2023_05_07', exist_ok=True)
            os.makedirs('data/TrustAIRLab/wild_jailbreak_2023_12_25', exist_ok=True)
            ds_test = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 'jailbreak_2023_05_07', split='train')
            ds_train = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 'jailbreak_2023_12_25', split='train')
            df_test_reg = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 'regular_2023_05_07', split='train')
            df_train_reg = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 'regular_2023_12_25', split='train')
            df_train_jail = ds_train.to_pandas()
            df_test_jail = ds_test.to_pandas()
            df_test_reg = df_test_reg.to_pandas()
            df_train_reg = df_train_reg.to_pandas()
            df_train_jail.drop_duplicates(subset=['prompt'], inplace=True)
            df_test_jail.drop_duplicates(subset=['prompt'], inplace=True)
            df_test_reg.drop_duplicates(subset=['prompt'], inplace=True)
            df_train_reg.drop_duplicates(subset=['prompt'], inplace=True)
            df_train_jail = df_train_jail[~df_train_jail['prompt'].isin(df_test_jail['prompt'])]
            df_train_reg = df_train_reg[~df_train_reg['prompt'].isin(df_test_reg['prompt'])]
            df_train_jail['split'] = 'train'
            df_test_jail['split'] = 'test'
            df_train_reg['split'] = 'train'
            df_test_reg['split'] = 'test'
            df_train_reg['is_malicious'] = 0
            df_test_reg['is_malicious'] = 0
            df_train_jail['is_malicious'] = 1
            df_test_jail['is_malicious'] = 1
            min_len_train = min(len(df_train_jail), len(df_train_reg))
            df_train = pd.concat([df_train_jail.sample(n=min_len_train, random_state=42), df_train_reg.sample(n=min_len_train, random_state=42)], ignore_index=True)
            min_len_test = min(len(df_test_jail), len(df_test_reg))
            df_test = pd.concat([df_test_jail.sample(n=min_len_test, random_state=42), df_test_reg.sample(n=min_len_test, random_state=42)], ignore_index=True)
            df = pd.concat([df_train.sample(frac=1, random_state=42), df_test], ignore_index=True)
            df.rename(columns={'prompt': 'query'}, inplace=True)
            df.drop_duplicates(subset=['query'], inplace=True)
            df.to_csv(os.path.join('data/TrustAIRLab', 'wild_jailbreak_combined.csv'), index=False)
        else:
            df = pd.read_csv(os.path.join('data/TrustAIRLab', 'wild_jailbreak_combined.csv'))
        if 'train' in split_name:
            df = df[df["split"]=='train']
        else:  # if 'test' in split_name:
            df = df[df["split"]=='test']
    return df


def get_pretrained_model(local_model_path: str, device: str, quantized: bool=True) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    It loads a 4‑bit quantized causal‑language model and its tokenizer.

    Parameters
    ----------
    local_model_path : str
        Path (or Hugging Face repo identifier) to where checkpoints are stored.
    device : str
        Target device for the model, e.g. ``"cpu"``, ``"cuda:0"``.
    quantized : bool, optional
        Whether to load the model with 4‑bit quantization, default ``True``.

    Returns
    -------
    tuple[AutoModelForCausalLM, AutoTokenizer]
        ``model`` – a `AutoModelForCausalLM` instance loaded with 4‑bit
        quantization (BitsAndBytes) and set to evaluation mode.  
        ``tokenizer`` – the corresponding tokenizer ready for encoding inputs.
    """
    # https://huggingface.co/blog/4bit-transformers-bitsandbytes
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                             bnb_4bit_compute_dtype=torch.bfloat16,
                                             bnb_4bit_use_double_quant=True,
                                             bnb_4bit_quant_type="nf4")
    print(f"Loading model from {local_model_path} with quantized={quantized}")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    
    if 't5gemma' in local_model_path.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(
            local_model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            quantization_config=quantization_config if quantized else None,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            quantization_config=quantization_config if quantized else None,
        )
    model.eval()
    return model, tokenizer


def prepare_input(model: torch.nn.Module, tokenizer,
                  queries: list[str], system_prompt: List[str]):
    batch_messages = []
    for query, prompt in zip(queries, system_prompt):
        if model.config.is_encoder_decoder:
            if prompt.strip() != "":
                messages = [{"role": "user", "content":  ''},  # 'Conversation roles must alternate user/assistant/user/assistant/...' 
                            {"role": "assistant", "content": prompt},
                            {"role": "user", "content":  query}]
            else:
                messages = [{"role": "user", "content":  query}]
        else:
            '''Causal models pick the logits of last token 
            to predict next token. If we did the padding on right hand side, 
            the model is actually using the logits of <PAD> to do next token sampling!
            So we pad left. 
            '''
            tokenizer.padding_side = "left"  # https://junronglin.com/article/why_left_padding
            if prompt.strip() != "":
                messages = [{"role": "system", "content": prompt},
                            {"role": "user", "content": query}]
            else:
                messages = [{"role": "user", "content":  query}]
        batch_messages.append(messages)
    
    return tokenizer.apply_chat_template(batch_messages,
                                         add_generation_prompt=True,
                                         return_tensors="pt",
                                         enable_thinking=False,
                                         return_dict=True,
                                         padding=True,
                                         max_length=512,
                                         truncation=True)


def compute_auc_pointbiserial(scores, labels, savefig_path: str):
    results_df = pd.DataFrame({'score': scores, 'label': labels})
    results_df['label'] = results_df['label'].astype(int)
    correlation, p_value = pointbiserialr(results_df['label'], results_df['score'])

    print(f"Point-biserial correlation: {correlation:.4f}")
    print(f"p-value: {p_value:.6f}")

    print('Mean score for label 0:', results_df[results_df['label'] == 0].score.mean())
    print('Mean score for label 1:', results_df[results_df['label'] == 1].score.mean())
    auc = roc_auc_score(results_df['label'], results_df['score'])
    print(f"AUC: {auc:.4f}")

    #print("AUC =", roc_auc_score(results_df['label'], results_df['score']))
    RocCurveDisplay.from_predictions(
        results_df['label'], 
        results_df['score'],
        name="Model",
        color="darkorange",
    )
    plt.plot([0,1], [0,1], color='navy', linestyle='--')
    plt.grid(True, alpha=0.3)
    if savefig_path is not None:
        plt.savefig(savefig_path)
    #plt.show()
    return auc


def compute_metrics_and_save(probas: List[float] = None,
                             labels: List[int] = None,
                             queries: List[str] = None,
                             save_folder: str = None,
                             threshold: float = None,
                             y_pred: List[int] = None,) -> pd.DataFrame:
    metrics = {}

    if probas is not None:
        auc = compute_auc_pointbiserial(probas, labels, os.path.join(save_folder, 'roc_curve.png'))
        metrics['auc'] = auc
        if threshold is not None and y_pred is None:  # If threshold is provided but y_pred is not, compute y_pred based on the threshold.
            y_pred = [1 if p >= threshold else 0 for p in probas]
        else:  # If threshold is not provided, but y_pred is, keep y_pred. Compute the optimal threshold for the ROC curve only.
            fpr, tpr, thresholds = roc_curve(labels, probas)
            dist = fpr ** 2 + (1 - tpr) ** 2
            best_idx = dist.argmin()
            threshold = float(thresholds[best_idx])
            y_pred = [1 if p >= threshold else 0 for p in probas] if y_pred is None else y_pred
        metrics['threshold'] = threshold

    report_json = classification_report(labels, y_pred,
                                        output_dict=True,
                                        target_names=['benign', 'malicious'])
    metrics['classification_report'] = report_json

    if save_folder is not None:
        with open(os.path.join(save_folder, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

    cm = confusion_matrix(labels, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['benign', 'malicious'])
    disp.plot(cmap='Blues')
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, 'confusion_matrix.png'))
    plt.close()

    results = {'query': queries, 'is_malicious': labels}
    if probas is not None:
        results['probas'] = probas
    results['y_pred'] = y_pred
    csv_name = 'raw_predictions.csv'
    results = pd.DataFrame(results)
    if save_folder is not None:
        results.to_csv(os.path.join(save_folder, csv_name), index=False)
    return results
