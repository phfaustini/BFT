import os
import gc
from typing import List, Tuple
from pickle import dump, load
from joblib import Parallel, delayed

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import detection.models_datasets as md


def _hook_fn_to_monitor_attention(module, input, output):
    # Registered on self_attn.o_proj, so input[0] == softmax(QK^T / √d) @ V,
    # concatenated across heads with layout [head_0_dim_0..head_0_dim_{d-1}, head_1_dim_0, ...].
    # Shape: [batch, seq_len, n_heads * head_dim]. HF makes this contiguous before o_proj, so view() is valid.
    attn_output = input[0].detach()
    batch, seq_len, _ = attn_output.shape
    attn_output = attn_output.view(batch, seq_len, module.n_heads, -1)

    last_token_attentions = attn_output[:, -1, :, :]  # [batch, n_heads, head_dim]

    is_pure_zero = (module.attn_outputs_last_token == 0).all(dim=(1, 2))
    first_zero_idx = is_pure_zero.nonzero(as_tuple=True)[0][0].item()
    end_idx = min(first_zero_idx + batch, module.attn_outputs_last_token.shape[0])
    module.attn_outputs_last_token[first_zero_idx:end_idx] = last_token_attentions.cpu()
    return None


def get_attention_activations(model, tokenizer,
                              prompts: List[str], system_prompts: List[str],
                              batch_size: int=50,
                              target_layers: List[int]=None) -> torch.Tensor:
    """It returns the attention activations for the last token in each prompt.
    Attention activations are of shape [n_sentences, n_layers, n_heads_per_layer, seq_len_last_token]
    → i.e. [n_sentences, n_layers, n_heads, dimensionality]

    Parameters:
    ----------
        model: The language model.
        tokenizer: The tokenizer for the model.
        prompts (List[str]): The list of prompts.
        system_prompts (List[str]): The list of system prompts.
        batch_size (int): The batch size for inference.
        target_layers (List[int]): If provided, only hook these layers (saves memory at inference time).
            When None, all layers are hooked.

    Returns:
    ----------
        attentions_last_token (Tensor): [n_sentences, n_layers, n_heads_per_layer, dimensionality]
    """
    model.eval()
    encodings = md.prepare_input(model, tokenizer, prompts, system_prompts)  #tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
    input_ids = encodings.input_ids
    attention_mask = encodings.attention_mask
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if model.config.is_encoder_decoder:
        handles_encoder = []
        handles_decoder = []
        encoder_layers = model.model.encoder.layers
        decoder_layers = model.model.decoder.layers
        if target_layers is not None:
            encoder_layers = [encoder_layers[i] for i in target_layers if i < len(encoder_layers)]
            decoder_layers = [decoder_layers[i] for i in target_layers if i < len(decoder_layers)]
        for layer in encoder_layers:
            o_proj = layer.self_attn.o_proj
            n_heads = model.config.encoder.num_attention_heads
            o_proj.n_heads = n_heads
            o_proj.attn_outputs_last_token = torch.zeros(input_ids.shape[0],
                                                         n_heads,
                                                         o_proj.in_features // n_heads,
                                                         device='cpu', dtype=torch.bfloat16)
            handle_encoder = o_proj.register_forward_hook(_hook_fn_to_monitor_attention)
            handles_encoder.append(handle_encoder)
        for layer in decoder_layers:
            o_proj = layer.self_attn.o_proj
            n_heads = model.config.decoder.num_attention_heads
            o_proj.n_heads = n_heads
            o_proj.attn_outputs_last_token = torch.zeros(input_ids.shape[0],
                                                         n_heads,
                                                         o_proj.in_features // n_heads,
                                                         device='cpu', dtype=torch.bfloat16)
            handle_decoder = o_proj.register_forward_hook(_hook_fn_to_monitor_attention)
            handles_decoder.append(handle_decoder)

        with torch.inference_mode():
            for batch_input_ids, batch_attention_mask in dataloader:
                outputs = model(
                    input_ids=batch_input_ids.to(model.device),
                    attention_mask=batch_attention_mask.to(model.device),
                    decoder_input_ids=batch_input_ids.clone().to(model.device),
                )
        [h.remove() for h in handles_encoder]
        [h.remove() for h in handles_decoder]
        attentions_last_token_encoder = torch.stack([layer.self_attn.o_proj.attn_outputs_last_token for layer in encoder_layers], dim=1)
        attentions_last_token_decoder = torch.stack([layer.self_attn.o_proj.attn_outputs_last_token for layer in decoder_layers], dim=1)
        return attentions_last_token_encoder, attentions_last_token_decoder
    elif hasattr(model.model, "get_decoder"):  # Gemma 3 4B
        handles = []
        all_layers = model.model.get_decoder().layers
        layers_to_hook = [all_layers[i] for i in target_layers] if target_layers is not None else all_layers
        decoder_cfg = model.model.get_decoder().config
        for layer in layers_to_hook:
            o_proj = layer.self_attn.o_proj
            n_heads = decoder_cfg.num_attention_heads
            o_proj.n_heads = n_heads
            o_proj.attn_outputs_last_token = torch.zeros(input_ids.shape[0],
                                                         n_heads,
                                                         o_proj.in_features // n_heads,
                                                         device='cpu', dtype=torch.bfloat16)
            handle = o_proj.register_forward_hook(_hook_fn_to_monitor_attention)
            handles.append(handle)

        with torch.inference_mode():
            for batch_input_ids, batch_attention_mask in dataloader:
                outputs = model(
                    input_ids=batch_input_ids.to(model.device),
                    attention_mask=batch_attention_mask.to(model.device),
                )
        [h.remove() for h in handles]
        attentions_last_token = torch.stack([layer.self_attn.o_proj.attn_outputs_last_token for layer in layers_to_hook], dim=1)
        return attentions_last_token
    else:
        handles = []
        all_layers = model.model.layers
        layers_to_hook = [all_layers[i] for i in target_layers] if target_layers is not None else all_layers
        for layer in layers_to_hook:
            o_proj = layer.self_attn.o_proj
            n_heads = model.config.num_attention_heads
            o_proj.n_heads = n_heads
            o_proj.attn_outputs_last_token = torch.zeros(input_ids.shape[0],
                                                         n_heads,
                                                         o_proj.in_features // n_heads,
                                                         device='cpu', dtype=torch.bfloat16)
            handle = o_proj.register_forward_hook(_hook_fn_to_monitor_attention)
            handles.append(handle)
        print(" Done register_forward_hook")
        with torch.inference_mode():
            for batch_input_ids, batch_attention_mask in dataloader:
                outputs = model(
                    input_ids=batch_input_ids.to(model.device),
                    attention_mask=batch_attention_mask.to(model.device),
                )
        print(" Completed inference for all batches.")
        [h.remove() for h in handles]
        attentions_last_token = torch.stack([layer.self_attn.o_proj.attn_outputs_last_token for layer in layers_to_hook], dim=1)
        return attentions_last_token


def build_datasets(attentions: Tensor, labels: Tensor) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """It creates vectors from the attention activations for each (layer, head) pair.
    It then returns n_layers * n_heads_per_layer datasets. So each dataset corresponds to
    the activations of a specific (layer, head) across all sentences.

    Parameters:
    ----------
        attentions (Tensor): [n_sentences, n_layers, n_heads_per_layer, dimensionality]
        labels (Tensor): [n_sentences]

    Returns:
    ----------
        Xs (np.ndarray): n_layers * n_heads_per_layer datasets with shape [n_sentences, dimensionality]
        Ys (np.ndarray): n_layers * n_heads_per_layer datasets with shape [n_sentences]
    """
    n_layers = attentions.shape[1]
    n_heads_per_layer = attentions.shape[2]

    Xs = []
    Ys = []
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads_per_layer):
            dataset = attentions[:, layer_idx, head_idx, :]  # shape n_sentences, dimensionality
            Xs.append(dataset.detach().to(dtype=torch.float32).cpu().float().numpy())
            Ys.append(labels.detach().cpu().float().numpy())

    return Xs, Ys


def create_probe(X: np.ndarray, y: np.ndarray,
                 clf_name: str ='LogisticRegression',
                 random_state: int=42) -> Tuple[LogisticRegression, float]:
    """Create and train a LogisticRegression regression probe on the given data.
    Following the paper, we use half of the data for training and half for testing.

    The probe predicts binary labels (0 or 1), so after predicting
    we threshold the outputs at 0.5 to get binary predictions.
    We then compute the F1 score on the test set.

    Parameters:
    ----------
        X (np.ndarray): Input features of shape [n_samples, n_features].
        y (np.ndarray): Target labels of shape [n_samples]. Binary classification (0 or 1).
        clf_name (str): Classifier type. Default to 'LogisticRegression', but others are supported.
    
    Returns:
    ----------
        clf (LogisticRegression): Trained LogisticRegression model.
        f1 (float): F1 score on the test set.
    """
    if clf_name == 'LogisticRegression':
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=random_state, max_iter=1000))
        ])
    elif clf_name == 'SVM':
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(random_state=random_state, probability=True))
        ])
    else: # RandomForest
        clf = RandomForestClassifier(random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    f1 = f1_score(y_test, y_pred_binary)
    return clf, f1


def rank_probes(n_layers: int,
                n_heads_per_layer: int,
                Xs: List[np.ndarray],
                Ys: List[np.ndarray],
                clf_name: str ='LogisticRegression',
                n_jobs: int = 2,
                random_state: int=42) -> pd.DataFrame:
    """Create and evaluate probes for each (layer, head) pair.
    It returns a DataFrame with the layer index, head index, F1 score, and trained classifier.

    Parameters:
    ----------
        n_layers (int): Number of layers in the model.
        n_heads_per_layer (int): Number of heads per layer in the model.
        Xs (List[np.ndarray]): List of datasets for each (layer, head) pair.
        Ys (List[np.ndarray]): List of labels for each (layer, head) pair.
        clf_name (str): Classifier type. Default to 'LogisticRegression', but others are supported.
        n_jobs (int): Number of parallel jobs to use. Default is -1 (use all available cores).

    Returns:
    ----------
        interventions (pd.DataFrame): DataFrame with columns ["layer", "head", "f1", "clf"].
    """    
    def _process_probe(layer_idx, head_idx, X, y):
        clf, f1_value = create_probe(X, y, clf_name=clf_name, random_state=random_state)
        return {"layer": layer_idx, "head": head_idx, "f1": f1_value, "clf": clf}

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_probe)(
            layer_idx, head_idx, Xs[i], Ys[i]
        )
        for i, (layer_idx, head_idx) in enumerate(
            (l, h) for l in range(n_layers) for h in range(n_heads_per_layer)
        )
    )
    return pd.DataFrame(results)


def persist_probes(df: pd.DataFrame, probes_persistence_folder: str) -> None:
    """It saves the trained probes to disk.

    The probes are named according to their layer, head, and F1 score.

    Parameters:
    ----------
        df (pd.DataFrame): DataFrame with columns ["layer", "head", "f1", "clf"].
        filepath (str): Path to save the CSV file.
    """
    for _, row in df.iterrows():
        os.makedirs(probes_persistence_folder, exist_ok=True)
        with open(os.path.join(probes_persistence_folder, f"layer_{row['layer']}-head_{row['head']}-f1_{row['f1']:.5f}-probe.pkl"), "wb") as f:
            dump(row['clf'], f, protocol=5)


def load_probes(probes_persistence_folder: str) -> pd.DataFrame:
    """Load the DataFrame of probes from a CSV file
    (including the classifier objects).

    Parameters:
    ----------
        filepath (str): Path to load the CSV file.

    Returns:
    ----------
        pd.DataFrame: DataFrame with columns ["layer", "head", "f1", "clf"].
    """
    data = {"layer": [], "head": [], "f1": [], "clf": []}
    for filename in os.listdir(probes_persistence_folder):
        if filename.endswith(".pkl"):
            parts = filename.split("-")
            layer = int(parts[0].split("_")[1])
            head = int(parts[1].split("_")[1])
            f1 = float(parts[2].split("_")[1])
            with open(os.path.join(probes_persistence_folder, filename), "rb") as f:
                clf = load(f)
            data["layer"].append(layer)
            data["head"].append(head)
            data["f1"].append(f1)
            data["clf"].append(clf)
    assert len(data["clf"]) > 0, "No probes found in the specified folder."
    return pd.DataFrame(data)


def create_plot(df, filepath: str=None) -> None:
    """
    It creates a heatmap plot of F1 scores across layers and heads.

    Parameters:
    ----------
        df (pd.DataFrame): DataFrame with columns ["layer", "head", "f1"].
        filepath (str): If provided, save the plot to this file path.
    
    Returns:
    ----------
        None
    """
    # Pivot to create a 2D grid: rows = layers, cols = heads
    pivot_df = df.pivot_table(index='layer', columns='head', values='f1', aggfunc='mean')

    # Optional: sort heads by average F1 (to mimic "Head (Sorted)")
    #head_avg = df.groupby('head')['f1'].mean().sort_values(ascending=False)
    #pivot_df = pivot_df[head_avg.index]  # reorder columns

    # Optional: sort layers numerically (if not already)
    pivot_df = pivot_df.sort_index()  # assumes layer index is numeric

    # Extract as numpy array for plotting
    f1_matrix = pivot_df.values
    layers = pivot_df.index.tolist()
    heads = pivot_df.columns.tolist()

    plt.figure(figsize=(10, 6))

    # Display the matrix as an image
    im = plt.imshow(f1_matrix, 
                    cmap='plasma',       # or 'viridis', 'inferno', 'coolwarm', etc.
                    aspect='auto',       # stretch to fit axes
                    origin='upper')      # layer 0 (or min) at top

    # Invert y-axis if you want layer 0 at bottom (like typical plots)
    # plt.gca().invert_yaxis()

    # Set ticks and labels
    plt.yticks(ticks=np.arange(len(layers)), labels=layers)
    plt.xticks(ticks=np.arange(len(heads)), labels=heads)

    plt.xlabel('Head')
    plt.ylabel('Layer')
    plt.title('F1 Score Across Layers and Heads')

    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('F1 Score')

    # Optional: Rotate head labels if crowded
    # plt.xticks(rotation=90)

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()


def inference_probes_detection(probes_folder, save_folder, model_name, df_test, pos_label=1, batch_size=50, device=None, top_k=1):
    df_probes = load_probes(probes_folder)

    top_probes = df_probes.sort_values('f1', ascending=False).head(top_k)
    target_layers = sorted(top_probes['layer'].unique().tolist())
    layer_to_idx = {layer: idx for idx, layer in enumerate(target_layers)}

    """
    Say the top-3 probes are at layers [2, 5, 2] (two probes in layer 2, one in layer 5).

    target_layers becomes [2, 5] (sorted unique).

    The returned tensor from get_attention_activations has shape [n_sentences, 2, n_heads, dim], where dim 1 index 0 = layer 2, index 1 = layer 5.

    layer_to_idx is {2: 0, 5: 1}, so layer_to_idx[row['layer']] correctly maps the original layer number to the position in the tensor.
    """

    test_queries = df_test['query'].to_list()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = md.get_pretrained_model(model_name, device)
    attentions_last_token = get_attention_activations(model,
                                                      tokenizer,
                                                      test_queries,
                                                      [""] * len(test_queries),
                                                      batch_size,
                                                      target_layers=target_layers)

    all_probas = []
    for _, row in top_probes.iterrows():
        layer_idx = layer_to_idx[row['layer']]
        features = attentions_last_token[:, layer_idx, row['head'], :].reshape(len(test_queries), -1)
        features = features.detach().cpu().float().numpy()
        probas = row['clf'].predict_proba(features)
        all_probas.append(probas[:, pos_label])

    avg_probas = np.mean(all_probas, axis=0)
    md.compute_metrics_and_save(probas=avg_probas.tolist(),
                                labels=df_test['is_malicious'].to_list(),
                                queries=df_test['query'].to_list(),
                                save_folder=save_folder)
    return probes_folder


def run_probes_detection(model_name: str,
                         save_folder: str,
                         df_train: pd.DataFrame,
                         df_test: pd.DataFrame,
                         pos_label: int=1,
                         batch_size: int=50,
                         device=None,
                         top_k: int=1,
                         clf_name='LogisticRegression',
                         random_state: int=42):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    probes_folder = os.path.join(save_folder, "probes")
    if not os.path.exists(probes_folder) or len(os.listdir(probes_folder)) == 0:
        train_queries = df_train['query']
        train_labels = df_train['is_malicious'].astype(int)
        train_labels = torch.tensor(train_labels.values)

        attentions_train_path = os.path.join(save_folder, f"top_k_{top_k}", "attentions_last_token_train.pt")
        if os.path.exists(attentions_train_path):
            attentions_last_token = torch.load(attentions_train_path)
        else:
            os.makedirs(os.path.join(save_folder, f"top_k_{top_k}"), exist_ok=True)
            model, tokenizer = md.get_pretrained_model(model_name, device)
            print("Computing attention activations for training set...")
            attentions_last_token = get_attention_activations(model,
                                                              tokenizer,
                                                              train_queries,
                                                              [""] * len(train_queries),
                                                              batch_size)
            torch.save(attentions_last_token, attentions_train_path)
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            print("Done with training attention activations.")
        Xs, Ys = build_datasets(attentions_last_token, train_labels)
        #n_sentences = attentions_last_token.shape[0]
        #n_probes = attentions_last_token.shape[1] * attentions_last_token.shape[2]
        n_layers = attentions_last_token.shape[1]
        n_heads_per_layer = attentions_last_token.shape[2]
        del attentions_last_token
        torch.cuda.empty_cache()
        df_probes = rank_probes(n_layers, n_heads_per_layer, Xs, Ys, clf_name=clf_name, random_state=random_state)
        persist_probes(df_probes, probes_folder)
    else:
        df_probes = load_probes(probes_folder)
        os.makedirs(os.path.join(save_folder, f"top_k_{top_k}"), exist_ok=True)

    inference_probes_detection(probes_folder,
                               os.path.join(save_folder, f"top_k_{top_k}"),
                               model_name,
                               df_test,
                               pos_label,
                               batch_size,
                               device,
                               top_k=top_k)
    return probes_folder
