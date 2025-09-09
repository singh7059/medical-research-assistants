import numpy as np
from sklearn.neighbors import NearestNeighbors
from models.embeddings import embed_texts

class SimpleVectorStore:
    def __init__(self):
        self.docs = []
        self.embeddings = None
        self.nn = None  # NearestNeighbors model

    def add_documents(self, texts):
        embeddings = embed_texts(texts).cpu().numpy()
        self.docs.extend(texts)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        self.nn = NearestNeighbors(n_neighbors=3, metric='cosine')
        self.nn.fit(self.embeddings)

    # ✅ Correctly indented search method
    def search(self, query, top_k=3, similarity_threshold=0.5):
        if self.nn is None:
            return []

        query_embedding = embed_texts([query]).cpu().numpy()
        n_samples = len(self.docs)
        n_neighbors = min(top_k, n_samples)

        distances, indices = self.nn.kneighbors(query_embedding, n_neighbors=n_neighbors)
        
        # cosine distance to similarity
        similarities = 1 - distances[0]  
        relevant_docs = [self.docs[i] for i, sim in zip(indices[0], similarities) if sim >= similarity_threshold]

        return relevant_docs
