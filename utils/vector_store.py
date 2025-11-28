import os
import pickle
import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False


class VectorStore:
    """
    Simple vector store with cosine similarity search.
    If FAISS is installed, uses FAISS; otherwise uses numpy brute-force.
    """

    def __init__(self, dim=256, index_path="data/vector_index"):
        self.dim = dim
        self.index_path = index_path
        self.ids = []

        if _FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            self.index = None
            self.vectors = None

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def add(self, vectors, ids):
        arr = np.asarray(vectors, dtype="float32")
        arr = self._normalize(arr)
        if _FAISS_AVAILABLE:
            self.index.add(arr)
        else:
            if self.vectors is None:
                self.vectors = arr
            else:
                self.vectors = np.vstack([self.vectors, arr])
        self.ids.extend(ids)

    def search(self, query_vec, top_k=5):
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        q = self._normalize(q)
        results = []

        if _FAISS_AVAILABLE:
            D, I = self.index.search(q, top_k)
            for score, idx in zip(D[0], I[0]):
                if idx == -1:
                    continue
                results.append((self.ids[idx], float(score)))
        else:
            sims = (self.vectors @ q.T).squeeze()
            top_idx = np.argsort(-sims)[:top_k]
            for idx in top_idx:
                results.append((self.ids[idx], float(sims[idx])))

        return results

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(f"{self.index_path}.meta.pkl", "wb") as f:
            pickle.dump(self.ids, f)
        if _FAISS_AVAILABLE:
            faiss.write_index(self.index, f"{self.index_path}.faiss")
        else:
            with open(f"{self.index_path}.vectors.npy", "wb") as f:
                np.save(f, self.vectors)

    def load(self):
        meta_path = f"{self.index_path}.meta.pkl"
        if not os.path.exists(meta_path):
            return False
        with open(meta_path, "rb") as f:
            self.ids = pickle.load(f)
        if _FAISS_AVAILABLE:
            self.index = faiss.read_index(f"{self.index_path}.faiss")
        else:
            with open(f"{self.index_path}.vectors.npy", "rb") as f:
                self.vectors = np.load(f)
        return True
