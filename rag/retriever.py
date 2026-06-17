def retrieve(store, query_embedding, k=5):

    results = store.search_with_score(query_embedding, k)

    cleaned = []

    for text, score in results:
        # FAISS L2 distance — lower = more similar
        # Threshold 1.0 works well for normalized embeddings
        if text and score < 1.0 and text not in cleaned:
            cleaned.append(text)

    return cleaned