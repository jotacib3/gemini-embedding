"""Gemini Embedding 2: Multimodal Search Engine and Study Package.

A complete implementation of Google's Gemini Embedding 2 natively multimodal
embedding model paired with ChromaDB for cross-modal search.
"""

from gemini_embedding.config import settings
from gemini_embedding.embeddings import (
    embed_file,
    embed_merged_interleaved,
    embed_separate_batch,
    embed_text,
    file_kind,
    format_task_text,
    get_genai_client,
)
from gemini_embedding.indexer import index_directory
from gemini_embedding.sample_media import generate_sample_media
from gemini_embedding.search_engine import (
    search_by_image,
    search_by_text,
)
from gemini_embedding.vector_store import MultimodalVectorStore

__all__ = [
    "MultimodalVectorStore",
    "embed_file",
    "embed_merged_interleaved",
    "embed_separate_batch",
    "embed_text",
    "file_kind",
    "format_task_text",
    "generate_sample_media",
    "get_genai_client",
    "index_directory",
    "search_by_image",
    "search_by_text",
    "settings",
]
