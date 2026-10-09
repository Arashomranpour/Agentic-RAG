import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class FastEmbedder:
    """Local ONNX embeddings (no API key, no GPU, no torch)."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # imported lazily: heavy

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(list(self._model.embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class FakeEmbedder:
    """Deterministic hashed bag-of-words vectors, so tests need no model download."""

    DIM = 256

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.DIM, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.DIM
            vec[bucket] += 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def build_embedder(backend: str, model_name: str) -> Embedder:
    if backend == "fake":
        return FakeEmbedder()
    return FastEmbedder(model_name)
