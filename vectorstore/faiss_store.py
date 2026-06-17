import faiss
import numpy as np
import pickle


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

    # ── SAFE SERIALIZATION ──────────────────────────────
    # FAISS index objects are C++ objects wrapped via SWIG and are NOT
    # reliably picklable across different environments/versions
    # (this is exactly what crashed on Streamlit Cloud).
    # Fix: serialize the FAISS index using FAISS's own native format,
    # and pickle only the plain Python `texts` list separately.

    def save(self, path_prefix):
        """Save index + texts. Creates: {path_prefix}.index and {path_prefix}.texts.pkl"""
        faiss.write_index(self.index, f"{path_prefix}.index")
        with open(f"{path_prefix}.texts.pkl", "wb") as f:
            pickle.dump(self.texts, f)

    @classmethod
    def load(cls, path_prefix):
        """Load index + texts back into a VectorStore instance."""
        store = cls()
        store.index = faiss.read_index(f"{path_prefix}.index")
        with open(f"{path_prefix}.texts.pkl", "rb") as f:
            store.texts = pickle.load(f)
        return store
