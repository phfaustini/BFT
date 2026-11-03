import os
import gc
from typing import List, Tuple, Union
from pickle import dump, load
from joblib import Parallel, delayed

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

import detection.models_datasets as md


def get_hidden_activations(model, tokenizer,
                           prompts: List[str], system_prompts: List[str],
                           batch_size: int = 50,
                           target_layers: List[int] = None
                           ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Extract the last-token residual-stream hidden states for every transformer layer.

    Uses HuggingFace's built-in ``output_hidden_states=True`` interface, so no forward
    hooks are needed. For decoder-only models, the returned tuple has ``n_layers + 1``
    entries (embedding + each transformer block); we stack all of them along a new
    layer dimension.

    Parameters:
    ----------
        model: The language model.
        tokenizer: The tokenizer for the model.
        prompts (List[str]): The list of prompts.
        system_prompts (List[str]): The list of system prompts.
        batch_size (int): The batch size for inference.
        target_layers (List[int]): If provided, keep only these layer indices (after
            stacking). Useful at inference time to reduce memory.

    Returns:
    ----------
        For causal models: Tensor of shape [n_sentences, n_layers, hidden_dim].
        For encoder-decoder models: (encoder_hidden, decoder_hidden) with the same
        [n_sentences, n_layers, hidden_dim] layout.
    """
    model.eval()
    encodings = md.prepare_input(model, tokenizer, prompts, system_prompts)
    input_ids = encodings.input_ids
    attention_mask = encodings.attention_mask
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if model.config.is_encoder_decoder:
        encoder_chunks, decoder_chunks = [], []
        with torch.inference_mode():
            for batch_input_ids, batch_attention_mask in dataloader:
                outputs = model(
                    input_ids=batch_input_ids.to(model.device),
                    attention_mask=batch_attention_mask.to(model.device),
                    decoder_input_ids=batch_input_ids.clone().to(model.device),
                    output_hidden_states=True,
                    return_dict=True,
                )
                enc = torch.stack([h[:, -1, :] for h in outputs.encoder_hidden_states], dim=1)
                dec = torch.stack([h[:, -1, :] for h in outputs.decoder_hidden_states], dim=1)
                encoder_chunks.append(enc.detach().to(dtype=torch.bfloat16, device='cpu'))
                decoder_chunks.append(dec.detach().to(dtype=torch.bfloat16, device='cpu'))
        enc_all = torch.cat(encoder_chunks, dim=0)
        dec_all = torch.cat(decoder_chunks, dim=0)
        if target_layers is not None:
            enc_all = enc_all[:, target_layers, :]
            dec_all = dec_all[:, target_layers, :]
        return enc_all, dec_all

    all_chunks = []
    with torch.inference_mode():
        for batch_input_ids, batch_attention_mask in dataloader:
            outputs = model(
                input_ids=batch_input_ids.to(model.device),
                attention_mask=batch_attention_mask.to(model.device),
                output_hidden_states=True,
                return_dict=True,
            )
            last_token = torch.stack([h[:, -1, :] for h in outputs.hidden_states], dim=1)
            all_chunks.append(last_token.detach().to(dtype=torch.bfloat16, device='cpu'))
    print(" Completed inference for all batches.")
    hidden = torch.cat(all_chunks, dim=0)
    if target_layers is not None:
        hidden = hidden[:, target_layers, :]
    return hidden


def build_datasets(hidden: Tensor, labels: Tensor) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Create one dataset per layer from the stacked hidden states.

    Parameters:
    ----------
        hidden (Tensor): [n_sentences, n_layers, hidden_dim]
        labels (Tensor): [n_sentences]

    Returns:
    ----------
        Xs (List[np.ndarray]): n_layers arrays of shape [n_sentences, hidden_dim].
        Ys (List[np.ndarray]): n_layers arrays of shape [n_sentences].
    """
    n_layers = hidden.shape[1]
    Xs, Ys = [], []
    for layer_idx in range(n_layers):
        Xs.append(hidden[:, layer_idx, :].detach().to(dtype=torch.float32).cpu().float().numpy())
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
                Xs: List[np.ndarray],
                Ys: List[np.ndarray],
                clf_name: str = 'LogisticRegression',
                n_jobs: int = 2,
                random_state: int=42) -> pd.DataFrame:
    """Train one probe per layer and return a DataFrame ranked by layer.

    Returns:
    ----------
        df (pd.DataFrame): columns ["layer", "f1", "clf"].
    """
    def _process_probe(layer_idx, X, y):
        clf, f1_value = create_probe(X, y, clf_name=clf_name, random_state=random_state)
        return {"layer": layer_idx, "f1": f1_value, "clf": clf}

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_probe)(layer_idx, Xs[layer_idx], Ys[layer_idx])
        for layer_idx in range(n_layers)
    )
    return pd.DataFrame(results)


def persist_probes(df: pd.DataFrame, probes_persistence_folder: str) -> None:
    """Save the trained probes to disk, one file per layer."""
    os.makedirs(probes_persistence_folder, exist_ok=True)
    for _, row in df.iterrows():
        path = os.path.join(
            probes_persistence_folder,
            f"layer_{row['layer']}-f1_{row['f1']:.5f}-probe.pkl",
        )
        with open(path, "wb") as f:
            dump(row['clf'], f, protocol=5)


def load_probes(probes_persistence_folder: str) -> pd.DataFrame:
    """Load probes persisted by ``persist_probes``. Returns columns ["layer", "f1", "clf"]."""
    data = {"layer": [], "f1": [], "clf": []}
    for filename in os.listdir(probes_persistence_folder):
        if not filename.endswith(".pkl"):
            continue
        parts = filename.split("-")
        layer = int(parts[0].split("_")[1])
        f1 = float(parts[1].split("_")[1])
        with open(os.path.join(probes_persistence_folder, filename), "rb") as f:
            clf = load(f)
        data["layer"].append(layer)
        data["f1"].append(f1)
        data["clf"].append(clf)
    assert len(data["clf"]) > 0, "No probes found in the specified folder."
    return pd.DataFrame(data)


def create_plot(df, filepath: str=None) -> None:
    """Plot F1 score per layer for hidden-state probes.

    Parameters:
    ----------
        df (pd.DataFrame): DataFrame with columns ["layer", "f1"].
        filepath (str): If provided, save the plot to this file path.
    """
    df_sorted = df.sort_values('layer')
    layers = df_sorted['layer'].tolist()
    f1_values = df_sorted['f1'].tolist()

    plt.figure(figsize=(10, 6))
    plt.plot(layers, f1_values, marker='o', linestyle='-', color='C0')
    plt.xticks(layers)
    plt.xlabel('Layer')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Across Layers')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()


def inference_probes_detection(probes_folder, save_folder, model_name, df_test,
                               pos_label=1, batch_size=50, device=None, top_k=1):
    df_probes = load_probes(probes_folder)

    top_probes = df_probes.sort_values('f1', ascending=False).head(top_k)
    target_layers = sorted(top_probes['layer'].unique().tolist())
    layer_to_idx = {layer: idx for idx, layer in enumerate(target_layers)}

    test_queries = df_test['query'].to_list()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = md.get_pretrained_model(model_name, device)
    hidden = get_hidden_activations(model,
                                    tokenizer,
                                    test_queries,
                                    [""] * len(test_queries),
                                    batch_size,
                                    target_layers=target_layers)

    all_probas = []
    for _, row in top_probes.iterrows():
        layer_idx = layer_to_idx[row['layer']]
        features = hidden[:, layer_idx, :].detach().cpu().float().numpy()
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
                         pos_label: int = 1,
                         batch_size: int = 50,
                         device=None,
                         top_k: int = 1,
                         clf_name: str = 'LogisticRegression',
                         random_state: int=42):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    probes_folder = os.path.join(save_folder, "probes")
    if not os.path.exists(probes_folder) or len(os.listdir(probes_folder)) == 0:
        train_queries = df_train['query']
        train_labels = torch.tensor(df_train['is_malicious'].astype(int).values)

        hidden_train_path = os.path.join(save_folder, f"top_k_{top_k}", "hidden_last_token_train.pt")
        if os.path.exists(hidden_train_path):
            hidden = torch.load(hidden_train_path)
        else:
            os.makedirs(os.path.join(save_folder, f"top_k_{top_k}"), exist_ok=True)
            model, tokenizer = md.get_pretrained_model(model_name, device)
            print("Computing hidden activations for training set...")
            hidden = get_hidden_activations(model,
                                            tokenizer,
                                            train_queries,
                                            [""] * len(train_queries),
                                            batch_size)
            torch.save(hidden, hidden_train_path)
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            print("Done with training hidden activations.")
        Xs, Ys = build_datasets(hidden, train_labels)
        n_layers = hidden.shape[1]
        del hidden
        torch.cuda.empty_cache()
        df_probes = rank_probes(n_layers, Xs, Ys, clf_name=clf_name, random_state=random_state)
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
