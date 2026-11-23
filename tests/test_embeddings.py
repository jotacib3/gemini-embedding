"""Tests for Gemini Embedding 2 formatting and embedding generation."""

from pathlib import Path
from unittest.mock import MagicMock

from google.genai import types

from gemini_embedding.embeddings import (
    embed_file,
    embed_merged_interleaved,
    embed_separate_batch,
    embed_text,
    file_kind,
    format_task_text,
)


def test_format_task_text_asymmetric():
    # Query formatting
    q_search = format_task_text("a red car", task="search result", is_query=True)
    assert q_search == "task: search result | query: a red car"

    q_qa = format_task_text("how do embeddings work?", task="question answering", is_query=True)
    assert q_qa == "task: question answering | query: how do embeddings work?"

    # Document formatting with title
    doc_titled = format_task_text(
        "Description of car", task="search result", is_query=False, title="Red Car"
    )
    assert doc_titled == "title: Red Car | text: Description of car"

    # Document formatting without title (Google requires 'title: none')
    doc_untitled = format_task_text(
        "Description of car", task="search result", is_query=False, title=None
    )
    assert doc_untitled == "title: none | text: Description of car"


def test_format_task_text_symmetric():
    # Classification is symmetric (same for query and doc)
    c1 = format_task_text("Support ticket text", task="classification", is_query=True)
    c2 = format_task_text("Support ticket text", task="classification", is_query=False)
    assert c1 == "task: classification | query: Support ticket text"
    assert c2 == "task: classification | query: Support ticket text"

    # Sentence similarity
    s = format_task_text("Hello world", task="sentence similarity")
    assert s == "task: sentence similarity | query: Hello world"


def test_file_kind():
    assert file_kind("photo.jpg") == "image"
    assert file_kind("image.PNG") == "image"
    assert file_kind("song.mp3") == "audio"
    assert file_kind("sample.wav") == "audio"
    assert file_kind("paper.pdf") == "pdf"
    assert file_kind("clip.mp4") == "video"
    assert file_kind("file.unknown") == "unknown"


def test_embed_text_mocked():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 768
    mock_response.embeddings = [mock_embedding]
    mock_client.models.embed_content.return_value = mock_response

    vec = embed_text(
        client=mock_client,
        text="quantum physics",
        task="search result",
        use_prefix=True,
        dims=768,
    )

    assert len(vec) == 768
    mock_client.models.embed_content.assert_called_once()
    call_kwargs = mock_client.models.embed_content.call_args.kwargs
    assert call_kwargs["contents"] == "task: search result | query: quantum physics"
    assert isinstance(call_kwargs["config"], types.EmbedContentConfig)
    assert call_kwargs["config"].output_dimensionality == 768


def test_embed_file_mocked(tmp_path: Path):
    test_file = tmp_path / "test.png"
    test_file.write_bytes(b"\x89PNG\r\n\x1a\nfakeimagebytes")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.2] * 768
    mock_response.embeddings = [mock_embedding]
    mock_client.models.embed_content.return_value = mock_response

    vec = embed_file(
        client=mock_client,
        path_or_bytes=test_file,
        dims=768,
    )

    assert len(vec) == 768
    mock_client.models.embed_content.assert_called_once()
    call_kwargs = mock_client.models.embed_content.call_args.kwargs
    contents = call_kwargs["contents"]
    assert len(contents) == 1
    assert contents[0].inline_data.mime_type == "image/png"


def test_embed_merged_interleaved_mocked(tmp_path: Path):
    test_file = tmp_path / "dog.jpg"
    test_file.write_bytes(b"testdogimage")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.3] * 768
    mock_response.embeddings = [mock_embedding]
    mock_client.models.embed_content.return_value = mock_response

    vec = embed_merged_interleaved(
        client=mock_client,
        text="A dog running on beach",
        file_path=test_file,
        dims=768,
    )

    assert len(vec) == 768
    mock_client.models.embed_content.assert_called_once()
    call_kwargs = mock_client.models.embed_content.call_args.kwargs
    contents = call_kwargs["contents"]
    assert len(contents) == 2
    assert contents[0] == "A dog running on beach"
    assert contents[1].inline_data.mime_type == "image/jpeg"


def test_embed_separate_batch_mocked(tmp_path: Path):
    test_file = tmp_path / "photo.png"
    test_file.write_bytes(b"photobytes")

    mock_client = MagicMock()
    mock_response = MagicMock()
    emb1, emb2 = MagicMock(), MagicMock()
    emb1.values = [0.1] * 768
    emb2.values = [0.2] * 768
    mock_response.embeddings = [emb1, emb2]
    mock_client.models.embed_content.return_value = mock_response

    batch = ["first query text", test_file]
    results = embed_separate_batch(client=mock_client, items=batch, dims=768)

    assert len(results) == 2
    mock_client.models.embed_content.assert_called_once()
    call_kwargs = mock_client.models.embed_content.call_args.kwargs
    contents = call_kwargs["contents"]
    # Verify each item was wrapped in its own Content object to prevent accidental merging
    assert len(contents) == 2
    assert isinstance(contents[0], types.Content)
    assert isinstance(contents[1], types.Content)


def test_separate_batch_rejects_more_than_six_images(tmp_path):
    """The API accepts at most 6 images per request; fail fast with a clear message."""
    import pytest
    from PIL import Image

    from gemini_embedding.embeddings import embed_separate_batch

    paths = []
    for i in range(7):
        path = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8)).save(path)
        paths.append(path)

    with pytest.raises(ValueError, match="at most 6 images"):
        embed_separate_batch(client=None, items=paths)  # type: ignore[arg-type]
