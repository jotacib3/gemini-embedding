"""Tests for application settings and configuration."""

from gemini_embedding.config import Settings


def test_default_settings():
    s = Settings()
    assert s.gemini_embedding_model == "gemini-embedding-2"
    assert s.embedding_dimensions == 768
    assert s.collection_name == "multimodal_media"
    assert ".jpg" in s.SUPPORTED_MIME_TYPES
    assert ".mp3" in s.SUPPORTED_MIME_TYPES
    assert ".wav" in s.SUPPORTED_MIME_TYPES
    assert ".pdf" in s.SUPPORTED_MIME_TYPES


def test_valid_dimensions():
    s = Settings()
    assert 768 in s.VALID_DIMENSIONS
    assert 1536 in s.VALID_DIMENSIONS
    assert 3072 in s.VALID_DIMENSIONS
