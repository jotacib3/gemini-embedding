"""Tests for ChromaDB vector store wrapper."""

from pathlib import Path

from gemini_embedding.vector_store import MultimodalVectorStore


def test_vector_store_crud(tmp_path: Path):
    persist_dir = tmp_path / "test_chroma"
    store = MultimodalVectorStore(persist_dir=persist_dir, collection_name="test_col")

    assert store.count() == 0

    # Upsert two items (simulated 4-dim vectors for testing)
    vec1 = [0.1, 0.2, 0.3, 0.4]
    vec2 = [0.9, 0.8, 0.7, 0.6]

    store.upsert_item(
        item_id="item-1",
        embedding=vec1,
        metadata={"filename": "photo.png", "kind": "image", "path": "/media/photo.png"},
    )
    store.upsert_item(
        item_id="item-2",
        embedding=vec2,
        metadata={"filename": "music.wav", "kind": "audio", "path": "/media/music.wav"},
    )

    assert store.count() == 2

    # Query with identical vector to vec1
    results = store.query(query_embedding=vec1, n_results=2)
    assert len(results) == 2
    assert results[0]["id"] == "item-1"
    assert results[0]["kind"] == "image"
    # Exact match in cosine distance is ~0.0, so similarity ~1.0
    assert results[0]["similarity"] >= 0.99

    # Filter query by kind
    audio_results = store.query(query_embedding=vec1, n_results=2, kind="audio")
    assert len(audio_results) == 1
    assert audio_results[0]["id"] == "item-2"
    assert audio_results[0]["kind"] == "audio"

    # Test clear
    store.clear()
    assert store.count() == 0
