import os
import math
from typing import List, Tuple
from pickle import dump, load

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from sentence_transformers import SentenceTransformer

import detection.models_datasets as md


def fjd_scores(model, encoded, device, batch_size=64, temperature=1.25) -> np.ndarray:
    """
    This function implements the FDJD detection attack described in https://github.com/GuoruiC/FJD

    Parameters:
    ----------
    model : torch.nn.Module
        The pre-loaded language model (e.g., a HuggingFace `AutoModelForCausalLM`).
    encoded : dict
        A dictionary containing the tokenized input ids and attention masks.
    device : str
        Target device identifier (e.g., ``"cpu"``, ``"cuda:0"``). 
    batch_size : int, optional
        Number of queries to process in a single batch. Default is ``64``.
    temperature : float, optional
        Sampling temperature for generation. Default is ``1.25``.

    Returns:
    -------
    np.ndarray
        An array of FJD scores for each input in the encoded dataset.
    """
    model.eval()
    model.to(device)
    dataset = TensorDataset(encoded['input_ids'], encoded['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_transition_scores = []
    with torch.inference_mode():
        for batch_input_ids, batch_attention_mask in dataloader:
            output = model.generate(
                input_ids=batch_input_ids.to(device),
                attention_mask=batch_attention_mask.to(device),
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
            )
            
            transition_scores = model.compute_transition_scores(
                output.sequences, 
                output.scores, 
                normalize_logits=True
            )
            all_transition_scores.append(transition_scores.cpu().float().numpy())
    
    return np.exp(np.concatenate(all_transition_scores, axis=0))


def llama_promptguard_detection(df_test: pd.DataFrame,
                                save_folder: str,
                                model_id = "meta-llama/Llama-Prompt-Guard-2-86M",
                                batch_size: int = 8):
    clf = pipeline("text-classification", model=model_id, batch_size=batch_size)
    queries = df_test['query'].to_list()
    is_malicious = df_test['is_malicious'].to_list()
    preds = clf(queries)
    y_pred = [1 if pred['label'] == 'LABEL_1' else 0 for pred in preds]
    probas = [pred['score'] if pred['label'] == 'LABEL_1' else 1 - pred['score'] for pred in preds]
    return md.compute_metrics_and_save(probas=probas, y_pred=y_pred, labels=is_malicious, queries=queries, save_folder=save_folder)    


def llama_guard_detection(df_test: pd.DataFrame,
                          save_folder: str,
                          model_id = "meta-llama/Llama-Guard-3-8B",
                          batch_size: int = 8):
    if "prompt" in model_id.lower():
        return llama_promptguard_detection(df_test, save_folder, model_id, batch_size)

    queries = df_test['query'].to_list()
    is_malicious = df_test['is_malicious'].to_list()

    """
    When queries have very different lengths, shorter ones
      get padded to the longest in the batch, wasting GPU 
      memory on padding tokens. 
      Sorting groups similar lengths together, 
      reducing peak memory per batch.
    """
    sorted_indices = sorted(range(len(queries)), key=lambda i: len(queries[i]))
    queries_sorted = [queries[i] for i in sorted_indices]

    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                             bnb_4bit_compute_dtype=torch.bfloat16,
                                             bnb_4bit_use_double_quant=True,
                                             bnb_4bit_quant_type="nf4")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto",
                                                 attn_implementation="eager",
                                                 quantization_config=quantization_config
                                                 )
    model.eval()
    y_pred = []
    with torch.inference_mode():
        for i in range(0, len(queries_sorted), batch_size):
            batch_queries = queries_sorted[i:i + batch_size]
            chats = [[{"role": "user", "content": q}] for q in batch_queries]
            inputs = tokenizer.apply_chat_template(
                chats, return_tensors="pt", padding=True, truncation=True,
                max_length=512, return_dict=True
            ).to(device)
            outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)

            for j, output in enumerate(outputs):
                prompt_len = inputs["input_ids"].shape[-1]
                result = tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
                y_pred.append(1 if "unsafe" in result.lower() else 0)
            del inputs, outputs
            torch.cuda.empty_cache()

    # Unsort predictions back to original order
    y_pred_original = [None] * len(y_pred)
    for sorted_pos, original_idx in enumerate(sorted_indices):
        y_pred_original[original_idx] = y_pred[sorted_pos]

    del model, tokenizer
    torch.cuda.empty_cache()
    return md.compute_metrics_and_save(y_pred=y_pred_original, labels=is_malicious, queries=queries, save_folder=save_folder)
     


def calculate_average_perplexity(queries: List[str], labels: List[int], device=None):
    """ GPT-NEO judges the perplexity of answers provided by another LLM.
    
    Parameters
    ----------
    queries : List[str]
        A list of query strings to be judged.    
    labels : List[int]
        A list of labels corresponding to each query.
    device : torch.device, optional
        The device to run the model on. If None, it will use GPU if available, else CPU.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A list of perplexity scores and labels corresponding to each query.
    """
    model_name = "EleutherAI/gpt-neo-2.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    perplexities = []
    ret_queries = []
    lbls = []
    for text, label in zip(queries, labels):
        try:
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = encodings.input_ids.to(device)
            attention_mask = encodings.attention_mask.to(device)
            
            with torch.inference_mode():
                outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
                loss = outputs.loss

            perplexity = math.exp(loss.item())
            if np.isnan(perplexity):
                continue
            perplexities.append(perplexity)
            lbls.append(1 if label else 0)
            ret_queries.append(text)
        except:
            continue
            #perplexities.append(float('inf'))  # Assign infinite perplexity if there's an error
    
    return np.array(perplexities), ret_queries, np.array(lbls)


def convert_text_to_embeddings(queries: list[str], model_name: str="sentence-transformers/all-mpnet-base-v2", batch_size: int=128, max_length: int=512) -> torch.Tensor:
    def truncate_queries(queries, model, max_length):
        truncated_queries = []
        for query in queries:
            tokens = model.tokenizer.encode(str(query), truncation=True, max_length=max_length, return_tensors='pt')
            truncated_query = model.tokenizer.decode(tokens[0], skip_special_tokens=True)
            truncated_queries.append(truncated_query)
        return truncated_queries

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=DEVICE, model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "eager"})
    model.eval()
    if "qwen" in model_name.lower():
        queries = truncate_queries(queries, model, max_length)
    with torch.inference_mode():
        embeddings = model.encode(queries,
                                  batch_size=batch_size,
                                  convert_to_tensor=True)
    del model
    torch.cuda.empty_cache()
    return embeddings


def run_embedding_detection(save_folder: str,
                            df_train: pd.DataFrame,
                            df_test: pd.DataFrame,
                            batch_size: int=64,
                            pos_label: int=1,
                            model_name: str="sentence-transformers/all-mpnet-base-v2",
                            random_state: int=42):
    test_queries = df_test['query'].to_list()

    if os.path.exists(embeddings_train_path :=  os.path.join(save_folder, "embeddings_train.npy")):
        print("Loading precomputed training embeddings...")
        with open(embeddings_train_path, 'rb') as f:
            train_embeddings = np.load(f)
    else:
        print("Computing training embeddings...")
        train_queries = df_train['query'].to_list()
        train_embeddings = convert_text_to_embeddings(train_queries, model_name=model_name, batch_size=batch_size)
        train_embeddings = train_embeddings.cpu().float().numpy()
        with open(embeddings_train_path, 'wb') as f:
            np.save(f, train_embeddings)

    saved_model_path = os.path.join(save_folder, f"embeddings_lr_model.pkl")
    if not os.path.exists(saved_model_path):
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=random_state, max_iter=1000))
        ])
        clf.fit(train_embeddings, df_train['is_malicious'].to_list())
        with open(saved_model_path, "wb") as f:
            dump(clf, f, protocol=5)
    else:
        with open(saved_model_path, "rb") as f:
            clf = load(f)

    if os.path.exists(embeddings_test_path := os.path.join(save_folder, "embeddings_test.npy")):
        print("Loading precomputed test embeddings...")
        with open(embeddings_test_path, 'rb') as f:
            test_embeddings = np.load(f)
    else:
        print("Computing test embeddings...")
        test_embeddings = convert_text_to_embeddings(test_queries, model_name=model_name, batch_size=batch_size)
        test_embeddings = test_embeddings.cpu().float().numpy()
        with open(embeddings_test_path, 'wb') as f:
            np.save(f, test_embeddings)
    probas = clf.predict_proba(test_embeddings)

    df_results = md.compute_metrics_and_save(probas[:, pos_label],
                                             df_test['is_malicious'].to_list(),
                                             test_queries,
                                             save_folder)                
    return clf, df_results


def run_fjd(model_name: str,
            save_folder: str,
            df_test: pd.DataFrame,
            batch_size: int=256):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    queries = df_test['query'].to_list()
    is_malicious = df_test['is_malicious'].to_list()

    model, tokenizer = md.get_pretrained_model(model_name, DEVICE)
    encoded = md.prepare_input(model,
                               tokenizer,
                               queries, 
                               system_prompt=['You are a good Assistant. ']*len(queries))

    trans = fjd_scores(model, encoded, DEVICE, temperature=1.25, batch_size=batch_size)
    first_token_scores = trans[:, 0]

    md.compute_metrics_and_save(first_token_scores,
                                is_malicious,
                                queries,
                                save_folder)
    del encoded
    torch.cuda.empty_cache()


def run_perplexity(save_folder: str, df_test: pd.DataFrame, device=None):
    queries = df_test['query'].to_list()
    is_malicious = df_test['is_malicious'].to_list()

    perplexities, queries, labels = calculate_average_perplexity(queries, is_malicious, device=device)
    return md.compute_metrics_and_save(perplexities,
                                       labels,
                                       queries,
                                       save_folder)


def run_tfidf_logistic_regression(save_folder: str,
                                  df_train: pd.DataFrame,
                                  df_test: pd.DataFrame,
                                  pos_label: int=1,
                                  random_state: int=42) -> Tuple[Pipeline, pd.DataFrame]:
    saved_model_path = os.path.join(save_folder, f"tfidf_lr_model.pkl")
    if not os.path.exists(saved_model_path):
        pipeline_lr = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('scale', StandardScaler(with_mean=False)),
            ('clf', LogisticRegression(random_state=random_state, max_iter=1000))
        ])
        pipeline_lr.fit(df_train['query'], df_train['is_malicious'])
        with open(saved_model_path, "wb") as f:
            dump(pipeline_lr, f, protocol=5)
    else:
        with open(saved_model_path, "rb") as f:
            pipeline_lr = load(f)

    probas = pipeline_lr.predict_proba(df_test['query'].to_list())

    df_results = md.compute_metrics_and_save(probas[:, pos_label],
                                             df_test['is_malicious'],
                                             df_test['query'],
                                             save_folder)                
    return pipeline_lr, df_results
