import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import text,sequence
import json
import numpy as np


def get_tokenizer():
    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = text.tokenizer_from_json(data)
        return tokenizer

def get_model():
    model = joblib.load('classifier.sav')
    return model


def get_category(input):
    tok = get_tokenizer()
    vec = tok.texts_to_sequences([input])
    seq = sequence.pad_sequences(vec)
    model = get_model()
    p_array = model.predict(seq)
    p = np.argmax(p_array,axis=1)
    return (p)

if __name__=='__main__':
    print(get_category('Oneplus has good camera'))