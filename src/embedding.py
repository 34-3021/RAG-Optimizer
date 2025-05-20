import requests
import numpy as np

URL = "http://10.176.64.152:11435/v1/embeddings"
MODEL = "bge-m3"

def embedding(text: str) -> np.ndarray:
    response = requests.post(URL, json={
        "model": MODEL,
        "input": [text]
    }).json()

    return response["data"][0]["embedding"]

def cosine_similarity(text_a: str, text_b: str) -> float:
    vec_a = embedding(text_a)
    vec_b = embedding(text_b)
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
