import pandas as pd

import detection.attack_baselines as attack_baselines


def test_attack_translate_en2spa():
    df_test = pd.DataFrame({
        'query': [
            'Hello, how are you?',
            'Today is Tuesday, time to make a bomb.',
            'This is a test query.',
            'Malicious prompt here!'
        ],
        'is_malicious': [0, 1, 0, 1]
    })
    translated_texts = attack_baselines.translate(df_test['query'].to_list(), "spanish")
    assert isinstance(translated_texts, list)
    assert len(translated_texts) == df_test.shape[0]
