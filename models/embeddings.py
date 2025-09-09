from sentence_transformers import SentenceTransformer

def get_embedding_model():
    # Small, free, fast model
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    model = get_embedding_model()
    return model.encode(texts, convert_to_tensor=True)
