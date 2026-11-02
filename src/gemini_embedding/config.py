"""Configuration management for Gemini Embedding 2 application."""

from pathlib import Path
from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    gemini_api_key: str | None = None

    # Model settings
    gemini_embedding_model: str = "gemini-embedding-2"
    embedding_dimensions: int = 768  # MRL: 768, 1536, or 3072

    # Storage paths
    chroma_persist_directory: str = "./chroma_db"
    default_media_dir: str = "./media"
    collection_name: str = "multimodal_media"

    # Supported MIME types according to Gemini Embedding 2 specification
    SUPPORTED_MIME_TYPES: ClassVar[dict[str, str]] = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".pdf": "application/pdf",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
    }

    # Recommended MRL dimensions
    VALID_DIMENSIONS: ClassVar[tuple[int, ...]] = (768, 1536, 3072)

    def get_chroma_path(self) -> Path:
        """Return the resolved ChromaDB persist directory."""
        return Path(self.chroma_persist_directory).resolve()

    def get_media_path(self) -> Path:
        """Return the resolved media directory."""
        return Path(self.default_media_dir).resolve()


settings = Settings()
