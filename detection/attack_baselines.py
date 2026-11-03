import pandas as pd
import torch
from transformers import pipeline


def translate(test_queries: list[str], target_language: str) -> list[str]:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    if target_language.lower() == "spanish":
        model_id = "Helsinki-NLP/opus-mt-tc-big-en-es"
    elif target_language.lower() == "chinese":
        model_id = "Helsinki-NLP/opus-mt-en-zh"
    elif target_language.lower() == "portuguese":
        model_id = "Helsinki-NLP/opus-mt-tc-big-en-pt"
        test_queries = [">>por<< " + q for q in test_queries]
    elif target_language.lower() == "macedonian":
        model_id = "Helsinki-NLP/opus-mt-synthetic-en-mk"
    elif target_language.lower() == "somali":
        model_id = "Helsinki-NLP/opus-mt-synthetic-en-so"
    pipe = pipeline("translation", model=model_id,
                    device=DEVICE, batch_size=32)
    translated = pipe(test_queries, truncation=True, do_sample=False, num_beams=1)
    translated = [t['translation_text'] for t in translated]
    return translated
