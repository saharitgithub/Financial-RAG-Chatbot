import faiss
import numpy as np


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