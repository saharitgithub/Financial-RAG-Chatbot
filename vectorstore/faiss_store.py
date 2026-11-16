import json
import os

import faiss
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VectorStore:

    def __init__(self):
        self.index = faiss.IndexFlatL2(384)
        self.texts = []

    def add(self, embedding, text):
        self.index.add(np.array([embedding]))
        self.texts.append(text)

    def search(self, query_embedding, k=5):

        distances, indices = self.index.search(
            np.array([query_embedding]), k
        )

        results = []

        for idx in indices[0]:

            if idx != -1:
                results.append(self.texts[idx])

        return results

    def search_with_score(self, query_embedding, k=5):

        distances, indices = self.index.search(
            np.array([query_embedding]), k
        )

        results = []

        for dist, idx in zip(distances[0], indices[0]):

            if idx != -1:
                results.append((self.texts[idx], float(dist)))

        return results

    def save(self, folder="store"):
        if not os.path.isabs(folder):
            folder = os.path.join(_ROOT, folder)
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "faiss.index"))
        with open(os.path.join(folder, "texts.json"), "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False)

    @classmethod
    def load(cls, folder="store"):
        if not os.path.isabs(folder):
            folder = os.path.join(_ROOT, folder)
        index_path = os.path.join(folder, "faiss.index")
        texts_path = os.path.join(folder, "texts.json")
        for p in (index_path, texts_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing store file: {p}")
        store = cls()
        store.index = faiss.read_index(index_path)
        with open(texts_path, "r", encoding="utf-8") as f:
            store.texts = json.load(f)
        return store