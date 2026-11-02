"""Tests for the synthetic sample media generator."""

import wave
from pathlib import Path

from PIL import Image

from gemini_embedding.sample_media import generate_sample_media


def test_generate_sample_media(tmp_path: Path):
    media_dir = tmp_path / "media"
    res = generate_sample_media(output_dir=media_dir)

    assert len(res["images"]) == 6
    assert len(res["audio"]) == 2
    assert len(res["queries"]) == 1

    # Check images validity
    for img_path in res["images"]:
        assert img_path.exists()
        with Image.open(img_path) as im:
            assert im.size == (512, 512)
            assert im.format in ("PNG", "JPEG")

    # Check query image
    query_img = res["queries"][0]
    assert query_img.exists()
    with Image.open(query_img) as im:
        assert im.size == (512, 512)

    # Check audio validity
    for aud_path in res["audio"]:
        assert aud_path.exists()
        with wave.open(str(aud_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 22050
            assert wf.getnframes() > 0
