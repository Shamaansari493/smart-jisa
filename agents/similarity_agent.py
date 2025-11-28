# agents/similarity_agent.py
import json
import os
from typing import List, Dict, Any

from agents.embedding_agent import embeddingAgent, EMBED_DIM
from utils.vector_store import VectorStore

DATA_PATH = "data/jira_issues.json"
INDEX_PATH = "data/vector_index"


class SimilarityAgent:
    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        self.store = VectorStore(dim=dim, index_path=INDEX_PATH)
        loaded = self.store.load()
        self.issues_by_key: Dict[str, Dict[str, Any]] = {}
        if not loaded:
            self._build_index()
        else:
            self._load_issues_into_memory()

    def _load_issues_list(self) -> List[Dict[str, Any]]:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_index(self):
        issues = self._load_issues_list()
        vectors = []
        ids = []
        for it in issues:
            key = it.get("issue_key") or it.get("id")
            text = f"{it.get('summary','')} . {it.get('description','')}"
            vec = embeddingAgent(text)
            vectors.append(vec)
            ids.append(key)
        self.store.add(vectors, ids)
        self.store.save()
        self.issues_by_key = {k: v for k, v in zip(ids, issues)}

    def _load_issues_into_memory(self):
        issues = self._load_issues_list()
        ids = [it.get("issue_key") or it.get("id") for it in issues]
        self.issues_by_key = {k: v for k, v in zip(ids, issues)}

    def find_similar(self, cleaned_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        qvec = embeddingAgent(cleaned_text)
        hits = self.store.search(qvec, top_k=top_k)
        results = []
        for issue_key, score in hits:
            item = self.issues_by_key.get(issue_key, {})
            results.append(
                {
                    "issue_key": issue_key,
                    "score": float(score),
                    "summary": item.get("summary"),
                    "description": item.get("description"),
                    "components": item.get("components", []),
                }
            )
        return results
