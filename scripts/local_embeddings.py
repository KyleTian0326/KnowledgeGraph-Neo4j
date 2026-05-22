import hashlib
import math
import os
import re
from pathlib import Path
from typing import Any

from neo4j_graphrag.embeddings.base import Embedder


DEFAULT_QWEN_MODEL_PATH = r"C:\Users\Kyle\.cache\modelscope\hub\models\Qwen\Qwen3-Embedding-0___6B"


class HashingEmbedder(Embedder):
    """Small local embedder for demos.

    It implements Neo4j GraphRAG's Embedder interface without calling a remote
    embedding API. For production semantic retrieval, replace this with a real
    embedding model such as BGE, sentence-transformers, OpenAI embeddings, etc.
    """

    def __init__(self, dimensions: int = 384):
        super().__init__()
        self.dimensions = dimensions

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _tokens(self, text: str) -> list[str]:
        lowered = text.lower()
        tokens = re.findall(r"[a-zA-Z0-9_]+", lowered)
        chinese = re.findall(r"[\u4e00-\u9fff]+", text)
        for segment in chinese:
            tokens.extend(segment[index : index + 2] for index in range(max(1, len(segment) - 1)))
            tokens.extend(segment[index : index + 3] for index in range(max(1, len(segment) - 2)))
        return [token for token in tokens if token.strip()]


class QwenLocalEmbedder(Embedder):
    """Local sentence-transformers embedder for Qwen3-Embedding.

    This is the production-quality local semantic embedder for this project.
    HashingEmbedder remains available as a dependency-free fallback.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_QWEN_MODEL_PATH,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ):
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for LOCAL_EMBEDDING_PROVIDER=qwen. "
                "Install it in this .venv or set LOCAL_EMBEDDING_PROVIDER=hashing."
            ) from exc

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Local Qwen embedding model not found: {model_path}")

        kwargs: dict[str, Any] = {}
        if device:
            kwargs["device"] = device
        self.model = SentenceTransformer(str(model_path), **kwargs)
        self.normalize_embeddings = normalize_embeddings
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimensions = int(self.model.get_embedding_dimension())
        else:
            self.dimensions = int(self.model.get_sentence_embedding_dimension())

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            text or "",
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.astype("float32").tolist()


def build_embedder() -> Embedder:
    provider = os.getenv("LOCAL_EMBEDDING_PROVIDER", "hashing").strip().lower()
    if provider in {"qwen", "qwen3", "sentence-transformers", "sentence_transformers"}:
        return QwenLocalEmbedder(
            model_path=os.getenv("LOCAL_EMBEDDING_MODEL_PATH", DEFAULT_QWEN_MODEL_PATH),
            device=os.getenv("LOCAL_EMBEDDING_DEVICE") or None,
            normalize_embeddings=os.getenv("LOCAL_EMBEDDING_NORMALIZE", "true").lower()
            not in {"0", "false", "no"},
        )
    if provider in {"hash", "hashing", "demo"}:
        return HashingEmbedder(dimensions=int(os.getenv("LOCAL_EMBEDDING_DIMENSIONS", "384")))
    raise ValueError(f"Unknown LOCAL_EMBEDDING_PROVIDER: {provider}")


def embedding_dimensions() -> int:
    provider = os.getenv("LOCAL_EMBEDDING_PROVIDER", "hashing").strip().lower()
    if provider in {"qwen", "qwen3", "sentence-transformers", "sentence_transformers"}:
        configured = os.getenv("LOCAL_EMBEDDING_DIMENSIONS")
        if configured:
            return int(configured)
        return 1024
    return int(os.getenv("LOCAL_EMBEDDING_DIMENSIONS", "384"))
