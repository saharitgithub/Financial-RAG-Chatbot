from ingestion.load_pdfs import load_pdfs
from ingestion.chunker import chunk_text
from embeddings.embedder import get_embedding
from vectorstore.faiss_store import VectorStore
import pickle

store = VectorStore()

docs = load_pdfs("data")
print(f"Total PDFs loaded: {len(docs)}")

total_chunks = 0

for doc in docs:
    chunks = chunk_text(doc)
    total_chunks += len(chunks)

    for c in chunks:
        emb = get_embedding(c)
        store.add(emb, c)

print(f"Total chunks indexed: {total_chunks}")
print("✅ FAISS INDEX BUILT SUCCESSFULLY")

with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)

print("✅ INDEX SAVED")