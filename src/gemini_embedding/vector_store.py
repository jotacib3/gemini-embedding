"""ChromaDB vector store integration for multimodal embeddings.

Configures ChromaDB with HNSW cosine similarity space to match Gemini
Embedding 2's representation space.
"""

from pathlib import Path
from typing import Any

import chromadb

from gemini_embedding.config import settings


class MultimodalVectorStore:
    """Manages persistent multimodal embeddings using ChromaDB."""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.persist_path = Path(persist_dir or settings.chroma_persist_directory)
        self.collection_name = collection_name or settings.collection_name
        self.persist_path.mkdir(parents=True, exist_ok=True)

        # Initialize persistent client to save vectors to disk
        self.client = chromadb.PersistentClient(path=str(self.persist_path))
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Any:
        """Create or load collection with cosine distance metric.

        IMPORTANT: ChromaDB defaults to L2 distance. Gemini embeddings must be
        evaluated with cosine similarity. We configure 'hnsw:space' = 'cosine'.
        """
        # Supports both ChromaDB 1.0+ configuration dict and legacy metadata
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_item(
        self,
        item_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Insert or update a single vector with its metadata."""
        self.collection.upsert(
            ids=[item_id],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def upsert_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Batch insert or update vectors."""
        if not ids:
            return
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 3,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query the vector store and return matched items with similarity scores.

        ChromaDB returns cosine distance (0 = identical, 2 = opposite).
        We convert it to cosine similarity: similarity = 1 - distance.
        """
        total = self.count()
        if total == 0:
            return []

        limit = min(n_results, total)
        where_clause = {"kind": kind} if kind else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_clause,
        )

        formatted_results: list[dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return formatted_results

        matched_ids = results["ids"][0]
        matched_metadatas = (
            results["metadatas"][0] if results.get("metadatas") else [{}] * len(matched_ids)
        )
        matched_distances = (
            results["distances"][0] if results.get("distances") else [0.0] * len(matched_ids)
        )

        for item_id, meta, dist in zip(matched_ids, matched_metadatas, matched_distances):
            # Distance to similarity: 1 - distance
            similarity = round(1.0 - float(dist), 3)
            formatted_results.append(
                {
                    "id": item_id,
                    "path": meta.get("path", ""),
                    "kind": meta.get("kind", ""),
                    "filename": meta.get("filename", Path(meta.get("path", "")).name),
                    "distance": round(float(dist), 4),
                    "similarity": similarity,
                    "metadata": meta,
                }
            )

        return formatted_results

    def count(self) -> int:
        """Return the total number of items indexed in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete all vectors in the collection and recreate it clean."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()
