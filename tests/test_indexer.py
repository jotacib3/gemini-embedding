"""Tests for media indexer and high-level search engine."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from gemini_embedding.indexer import index_directory
from gemini_embedding.sample_media import generate_sample_media
from gemini_embedding.search_engine import search_by_image, search_by_text
from gemini_embedding.vector_store import MultimodalVectorStore


def test_index_directory_and_search(tmp_path: Path):
    media_dir = tmp_path / "media"
    chroma_dir = tmp_path / "chroma"
    generate_sample_media(output_dir=media_dir)

    store = MultimodalVectorStore(persist_dir=chroma_dir, collection_name="test_index")

    mock_client = MagicMock()
    mock_vector = [0.05] * 768

    with patch("gemini_embedding.indexer.embed_file", return_value=mock_vector):
        indexed_count = index_directory(
            media_dir=media_dir,
            client=mock_client,
            vector_store=store,
            rate_limit_delay=0.0,
            show_progress=False,
        )

    # 6 images + 2 audios = 8 media files
    assert indexed_count == 8
    assert store.count() == 8

    # Test search_by_text with mocked embed_text
    with patch("gemini_embedding.search_engine.embed_text", return_value=mock_vector):
        results = search_by_text(
            query_text="red car",
            kind="image",
            client=mock_client,
            vector_store=store,
            n_results=3,
        )
        assert len(results) == 3
        for r in results:
            assert r["kind"] == "image"
            assert r["similarity"] >= 0.99

    # Test search_by_image with mocked embed_file
    query_image = media_dir / "car.png"
    with patch("gemini_embedding.search_engine.embed_file", return_value=mock_vector):
        results_img = search_by_image(
            image_path=query_image,
            kind="image",
            client=mock_client,
            vector_store=store,
            n_results=2,
        )
        assert len(results_img) == 2
        for r in results_img:
            assert r["kind"] == "image"
