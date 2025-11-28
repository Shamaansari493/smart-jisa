import numpy as np
import google.generativeai as genai

from utils.config import GOOGLE_API_KEY

EMBED_DIM = 768  # set based on model used; text-embedding-004 is 768, you can change if needed

genai.configure(api_key=GOOGLE_API_KEY)

def gemini_embed(text: str) -> np.ndarray:
    """
    Uses Gemini embeddings.
    Adjust model name if your course used a different one, e.g. 'models/text-embedding-004'.
    """
    resp = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
    )
    emb = resp["embedding"]
    return np.asarray(emb, dtype="float32")


def fallback_embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic hash-based embedding (offline fallback)."""
    import hashlib
    h = hashlib.sha256(text.encode("utf-8")).digest()
    arr = np.frombuffer(h, dtype=np.uint8).astype("float32")
    if arr.size >= dim:
        vec = arr[:dim]
    else:
        reps = int(np.ceil(dim / arr.size))
        vec = np.tile(arr, reps)[:dim]
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def embeddingAgent(text: str) -> np.ndarray:
    text = text or ""
    try:
        return gemini_embed(text)
    except Exception as e:
        print("Gemini embedding failed, falling back:", e)
        return fallback_embed(text)
