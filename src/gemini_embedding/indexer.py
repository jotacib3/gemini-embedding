"""Indexer module for populating ChromaDB with multimodal embeddings."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from google import genai
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from gemini_embedding.config import settings
from gemini_embedding.embeddings import embed_file, file_kind, get_genai_client
from gemini_embedding.vector_store import MultimodalVectorStore


def index_directory(
    media_dir: str | Path | None = None,
    client: genai.Client | None = None,
    vector_store: MultimodalVectorStore | None = None,
    rate_limit_delay: float = 0.5,
    show_progress: bool = True,
    progress_callback: Callable[[int, int, str], Any] | None = None,
) -> int:
    """Scan a media directory, generate embeddings, and upsert into ChromaDB.

    Args:
        media_dir: Path to directory containing media files.
        client: Optional Google GenAI client instance.
        vector_store: Optional MultimodalVectorStore instance.
        rate_limit_delay: Seconds to wait between API calls to stay within limits.
        show_progress: Whether to display a rich terminal progress bar.
        progress_callback: Optional hook for external consumers (e.g., notebooks).

    Returns:
        Number of indexed files.
    """
    source_dir = Path(media_dir or settings.get_media_path())
    if not source_dir.exists():
        raise FileNotFoundError(f"Media directory does not exist: {source_dir}")

    genai_client = client or get_genai_client()
    store = vector_store or MultimodalVectorStore()

    supported_suffixes = set(settings.SUPPORTED_MIME_TYPES.keys())
    files = sorted(
        p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in supported_suffixes
    )

    if not files:
        return 0

    total_files = len(files)

    if show_progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )
        task_id = progress.add_task("Indexing media...", total=total_files)
        progress.start()
    else:
        progress = None
        task_id = None

    try:
        for idx, file_path in enumerate(files):
            kind = file_kind(file_path)

            if progress and task_id is not None:
                progress.update(task_id, description=f"Embedding [{kind}] {file_path.name}")

            # Generate Gemini Embedding 2 representation
            vector = embed_file(
                client=genai_client,
                path_or_bytes=file_path,
                dims=settings.embedding_dimensions,
            )

            # Upsert into ChromaDB with metadata
            item_id = f"file-{idx:04d}"
            store.upsert_item(
                item_id=item_id,
                embedding=vector,
                metadata={
                    "path": str(file_path.resolve()),
                    "filename": file_path.name,
                    "kind": kind,
                    "suffix": file_path.suffix.lower(),
                },
            )

            if progress_callback:
                progress_callback(idx + 1, total_files, file_path.name)

            if progress and task_id is not None:
                progress.advance(task_id)

            # Stay under the rate limit
            if rate_limit_delay > 0 and idx < total_files - 1:
                time.sleep(rate_limit_delay)

    finally:
        if progress:
            progress.stop()

    return total_files
