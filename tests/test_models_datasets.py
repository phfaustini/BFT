import pytest
import torch

from detection.models_datasets import get_dataset, get_pretrained_model, prepare_input


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_get_dataset_renellm_wildguardmix():
    df_train = get_dataset("renellm_wildguardmix", "train")
    df_test = get_dataset("renellm_wildguardmix", "test")
    df_test_malicious = df_test[df_test['is_malicious']==1].copy()
    df_train_malicious = df_train[df_train['is_malicious']==1].copy()
    original_harm_behavior_train = set(df_train_malicious['original_harm_behavior'].unique().tolist())
    original_harm_behavior_test = set(df_test_malicious['original_harm_behavior'].unique().tolist())
    assert original_harm_behavior_train.isdisjoint(original_harm_behavior_test)

def test_get_dataset_autodanturbo_beavertails():
    df = get_dataset("autodanturbo_beavertails")
    assert df is not None
    assert "query" in df.columns
    assert "target" in df.columns
    assert "is_malicious" in df.columns
    assert "split" in df.columns
    assert set(df['is_malicious'].unique()) == {0, 1}
    assert df[df['is_malicious']==1].shape[0] == df[df['is_malicious']==0].shape[0]


def test_get_dataset_allenai_wildjailbreak():
    df = get_dataset("allenai_wildjailbreak")
    assert df is not None
    assert "query" in df.columns
    assert "target" in df.columns
    assert "is_malicious" in df.columns
    assert "split" in df.columns
    assert set(df['is_malicious'].unique()) == {0, 1}
    assert df[df['is_malicious']==1].shape[0] == df[df['is_malicious']==0].shape[0]


@pytest.mark.parametrize("dataset_name", ["BeaverTails", "TrustAIRLab"])
def test_get_test_dataset(dataset_name):
    df = get_dataset(dataset_name, split_name='330k_test')
    assert df is not None
    assert "query" in df.columns

@pytest.mark.parametrize("dataset_name", ["BeaverTails", "TrustAIRLab"])
def test_get_train_dataset(dataset_name):
    df = get_dataset(dataset_name, split_name='330k_train')
    assert df is not None
    assert "query" in df.columns
    assert "target" in df.columns if dataset_name != "TrustAIRLab" else "target" not in df.columns


causal_models = ["google/gemma-3-270m-it",
                 "meta-llama/Llama-3.2-1B-Instruct",
                 "Qwen/Qwen3-0.6B",]
@pytest.mark.parametrize("causal_model", causal_models)
def test_get_pretrained_model_causal(causal_model):
    model, tokenizer = get_pretrained_model(causal_model, DEVICE)
    assert model.config.is_encoder_decoder is False


def test_get_pretrained_model_encoderdecoder():
    model, tokenizer = get_pretrained_model("google/t5gemma-s-s-prefixlm-it", DEVICE)
    assert model.config.is_encoder_decoder is True


all_models = ["google/gemma-3-270m-it",
              "meta-llama/Llama-3.2-1B-Instruct",
              "Qwen/Qwen3-0.6B",
              "google/t5gemma-s-s-prefixlm-it"]
@pytest.mark.parametrize("model_name", all_models)
def test_prepare_input(model_name):
    model, tokenizer = get_pretrained_model(model_name, DEVICE)
    queries = ["Which is the closest planet to Earth that is not in the same galaxy as us?",
               "What is the capital of Panama?",
               "Where is the Sydney Opera House located?",
               "Who won the last FIFA World Cup?"]
    sp = ["" for _ in queries]
    inputs = prepare_input(model, tokenizer, queries, sp)
    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    decoded_strings = tokenizer.batch_decode(inputs['input_ids'])
    
    sp = ["Answer the question" for _ in queries]
    inputs = prepare_input(model, tokenizer, queries, sp)
    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    decoded_strings = tokenizer.batch_decode(inputs['input_ids'])
    assert all(["Answer the question" in ds for ds in decoded_strings])
