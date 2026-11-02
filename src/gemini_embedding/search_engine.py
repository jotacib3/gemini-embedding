"""High-level search orchestration for multimodal retrieval."""

from pathlib import Path
from typing import Any

from google import genai

from gemini_embedding.config import settings
from gemini_embedding.embeddings import embed_file, embed_text, get_genai_client
from gemini_embedding.vector_store import MultimodalVectorStore


def search_by_text(
    query_text: str,
    kind: str | None = None,
    use_prefix: bool = True,
    task: str = "search result",
    n_results: int = 3,
    client: genai.Client | None = None,
    vector_store: MultimodalVectorStore | None = None,
    dims: int | None = None,
) -> list[dict[str, Any]]:
    """Search indexed media using a natural language query.

    Args:
        query_text: The search description or prompt.
        kind: Optional modality filter ('image', 'audio', 'pdf', 'video').
        use_prefix: Whether to include the Gemini Embedding 2 task prefix.
        task: Asymmetric or symmetric task prefix identifier.
        n_results: Maximum number of closest matches to return.
        client: Optional Google GenAI client instance.
        vector_store: Optional MultimodalVectorStore instance.
        dims: Vector dimensions (defaults to configured MRL dimensions).

    Returns:
        List of result dictionaries containing id, path, kind, similarity, and metadata.
    """
    genai_client = client or get_genai_client()
    store = vector_store or MultimodalVectorStore()

    query_vector = embed_text(
        client=genai_client,
        text=query_text,
        task=task,
        use_prefix=use_prefix,
        dims=dims or settings.embedding_dimensions,
        is_query=True,
    )

    return store.query(
        query_embedding=query_vector,
        n_results=n_results,
        kind=kind,
    )


def search_by_image(
    image_path: str | Path,
    kind: str | None = "image",
    n_results: int = 3,
    client: genai.Client | None = None,
    vector_store: MultimodalVectorStore | None = None,
    dims: int | None = None,
) -> list[dict[str, Any]]:
    """Search indexed media using another image (Visual Search).

    Because Gemini Embedding 2 maps text and visual assets to the exact same
    latent vector space, an image embedding can query other images directly
    without relying on manual labels or text annotations.

    Args:
        image_path: Path to the query image (e.g. query.jpg).
        kind: Optional modality filter (typically 'image').
        n_results: Maximum number of closest matches.
        client: Optional Google GenAI client instance.
        vector_store: Optional MultimodalVectorStore instance.
        dims: Vector dimensions.

    Returns:
        List of result dictionaries.
    """
    genai_client = client or get_genai_client()
    store = vector_store or MultimodalVectorStore()

    query_vector = embed_file(
        client=genai_client,
        path_or_bytes=image_path,
        dims=dims or settings.embedding_dimensions,
    )

    return store.query(
        query_embedding=query_vector,
        n_results=n_results,
        kind=kind,
    )
