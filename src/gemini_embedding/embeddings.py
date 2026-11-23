"""Core embedding functions for Gemini Embedding 2.

Handles:
- Task prefix formatting for asymmetric and symmetric tasks
- Multimodal single-file embeddings (image, audio, PDF, video)
- Interleaved merged embeddings (text + image in one joint vector)
- Batch embeddings with distinct Content wrappers
- Matryoshka Representation Learning (MRL) dimension reduction
"""

import os
from collections.abc import Sequence
from pathlib import Path

from google import genai
from google.genai import types

from gemini_embedding.config import settings

# Official per-request limit for image inputs (PNG/JPEG).
MAX_IMAGES_PER_REQUEST = 6

# Supported task prefixes for Gemini Embedding 2.
# gemini-embedding-2 does not accept the `task_type` field: the task goes in the text itself.
ASYMMETRIC_TASKS = {
    "search result": "search result",
    "search": "search result",
    "question answering": "question answering",
    "qa": "question answering",
    "fact checking": "fact checking",
    "code retrieval": "code retrieval",
}

SYMMETRIC_TASKS = {
    "classification": "classification",
    "clustering": "clustering",
    "sentence similarity": "sentence similarity",
    "semantic similarity": "sentence similarity",
}


def get_genai_client(api_key: str | None = None) -> genai.Client:
    """Initialize and return a Google GenAI Client."""
    key = api_key or settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if key:
        return genai.Client(api_key=key)
    try:
        # Default Client attempts to pick up credentials or env vars
        return genai.Client()
    except Exception as err:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please configure GEMINI_API_KEY in your environment "
            "or create a .env file. Obtain a key at https://aistudio.google.com/"
        ) from err


def format_task_text(
    text: str,
    task: str = "search result",
    is_query: bool = True,
    title: str | None = None,
) -> str:
    """Format text with the official Gemini Embedding 2 task prefix.

    Gemini Embedding 2 embeds text differently depending on the objective:
    - Asymmetric tasks (Search, QA, Fact Checking, Code Retrieval):
        Query:    "task: <task> | query: <text>"
        Document: "title: <title> | text: <text>" (or "title: none | text: <text>")
    - Symmetric tasks (Classification, Clustering, Sentence Similarity):
        Both:     "task: <task> | query: <text>"

    Note: As recommended by Google, do NOT apply task prefixes to the text
    part of mixed multimodal inputs (e.g. caption + image).
    """
    clean_text = text.strip()
    task_lower = task.lower().strip()

    # Document side for asymmetric tasks
    if not is_query and task_lower in ASYMMETRIC_TASKS:
        doc_title = title.strip() if title else "none"
        return f"title: {doc_title} | text: {clean_text}"

    # Asymmetric query side
    if task_lower in ASYMMETRIC_TASKS:
        canonical_task = ASYMMETRIC_TASKS[task_lower]
        return f"task: {canonical_task} | query: {clean_text}"

    # Symmetric tasks
    if task_lower in SYMMETRIC_TASKS:
        canonical_task = SYMMETRIC_TASKS[task_lower]
        return f"task: {canonical_task} | query: {clean_text}"

    # Fallback to general search result prefix if custom task not recognized
    return f"task: {task_lower} | query: {clean_text}"


def file_kind(path: str | Path) -> str:
    """Determine the modality kind ('image', 'audio', 'pdf', 'video', or 'unknown')."""
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    mime = settings.SUPPORTED_MIME_TYPES.get(suffix, "")

    if not mime:
        return "unknown"
    if mime == "application/pdf":
        return "pdf"
    return mime.split("/")[0]


def embed_text(
    client: genai.Client,
    text: str,
    task: str = "search result",
    use_prefix: bool = True,
    dims: int | None = None,
    model: str | None = None,
    is_query: bool = True,
    title: str | None = None,
) -> list[float]:
    """Generate an embedding vector for a single text string."""
    target_dims = dims or settings.embedding_dimensions
    target_model = model or settings.gemini_embedding_model

    contents = (
        format_task_text(text, task=task, is_query=is_query, title=title) if use_prefix else text
    )

    config = types.EmbedContentConfig(output_dimensionality=target_dims)

    response = client.models.embed_content(
        model=target_model,
        contents=contents,
        config=config,
    )

    return response.embeddings[0].values


def embed_file(
    client: genai.Client,
    path_or_bytes: str | Path | bytes,
    mime_type: str | None = None,
    dims: int | None = None,
    model: str | None = None,
) -> list[float]:
    """Generate an embedding vector for a media file (image, audio, PDF, video)."""
    target_dims = dims or settings.embedding_dimensions
    target_model = model or settings.gemini_embedding_model

    if isinstance(path_or_bytes, bytes):
        data = path_or_bytes
        if not mime_type:
            raise ValueError("mime_type must be provided when passing raw bytes")
        resolved_mime = mime_type
    else:
        path_obj = Path(path_or_bytes)
        if not path_obj.is_file():
            raise FileNotFoundError(f"Media file not found: {path_obj}")
        data = path_obj.read_bytes()
        suffix = path_obj.suffix.lower()
        resolved_mime = mime_type or settings.SUPPORTED_MIME_TYPES.get(suffix)
        if not resolved_mime:
            raise ValueError(f"Unsupported media file type: {suffix}")

    part = types.Part.from_bytes(data=data, mime_type=resolved_mime)
    config = types.EmbedContentConfig(output_dimensionality=target_dims)

    response = client.models.embed_content(
        model=target_model,
        contents=[part],
        config=config,
    )

    return response.embeddings[0].values


def embed_merged_interleaved(
    client: genai.Client,
    text: str,
    file_path: str | Path,
    mime_type: str | None = None,
    dims: int | None = None,
    model: str | None = None,
) -> list[float]:
    """Demonstrate Gemini Embedding 2 interleaved aggregation behavior.

    Passing multiple parts (e.g. text + image bytes) in the contents list
    produces a single joint multimodal embedding that represents the combined
    concept (e.g. 'A photo of my dog at the beach' + beach_dog.jpg).
    """
    target_dims = dims or settings.embedding_dimensions
    target_model = model or settings.gemini_embedding_model

    path_obj = Path(file_path)
    data = path_obj.read_bytes()
    suffix = path_obj.suffix.lower()
    resolved_mime = mime_type or settings.SUPPORTED_MIME_TYPES.get(suffix)
    if not resolved_mime:
        raise ValueError(f"Unsupported media file type: {suffix}")

    image_part = types.Part.from_bytes(data=data, mime_type=resolved_mime)
    config = types.EmbedContentConfig(output_dimensionality=target_dims)

    # Note: Google advises NOT adding task prefixes to text in mixed multimodal inputs
    response = client.models.embed_content(
        model=target_model,
        contents=[text, image_part],
        config=config,
    )

    return response.embeddings[0].values


def embed_separate_batch(
    client: genai.Client,
    items: Sequence[str | Path],
    dims: int | None = None,
    model: str | None = None,
) -> list[list[float]]:
    """Embed multiple items and receive a separate vector for EACH item.

    Crucial distinction from Gemini Embedding 2:
    - contents=[part1, part2] -> Returns 1 MERGED embedding.
    - contents=[Content(parts=[part1]), Content(parts=[part2])] -> Returns 2 SEPARATE embeddings.
    """
    target_dims = dims or settings.embedding_dimensions
    target_model = model or settings.gemini_embedding_model

    image_count = sum(1 for item in items if file_kind(item) == "image")
    if image_count > MAX_IMAGES_PER_REQUEST:
        raise ValueError(
            f"Gemini Embedding 2 accepts at most {MAX_IMAGES_PER_REQUEST} images per request "
            f"(got {image_count}). Split the batch into smaller requests."
        )

    content_objects: list[types.Content] = []
    for item in items:
        if isinstance(item, str) and not Path(item).exists():
            # Text item
            content_objects.append(types.Content(parts=[types.Part.from_text(text=item)]))
        else:
            # File item
            path_obj = Path(item)
            suffix = path_obj.suffix.lower()
            mime = settings.SUPPORTED_MIME_TYPES.get(suffix)
            if not mime:
                raise ValueError(f"Unsupported file format: {suffix}")
            part = types.Part.from_bytes(data=path_obj.read_bytes(), mime_type=mime)
            content_objects.append(types.Content(parts=[part]))

    config = types.EmbedContentConfig(output_dimensionality=target_dims)
    response = client.models.embed_content(
        model=target_model,
        contents=content_objects,
        config=config,
    )

    return [emb.values for emb in response.embeddings]
