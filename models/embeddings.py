from sentence_transformers import SentenceTransformer

# Load model once (global)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    # Use the already loaded model
    return embedding_model.encode(texts, convert_to_tensor=True)
