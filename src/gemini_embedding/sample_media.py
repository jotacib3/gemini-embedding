"""Synthetic sample media generator.

Generates self-contained sample images, query images, and audio WAV files
so that the tutorial and multimodal search application can run immediately
out of the box without requiring external downloads.
"""

import math
import random
import struct
import wave
from pathlib import Path

from PIL import Image, ImageDraw

from gemini_embedding.config import settings


def create_gradient_image(
    filename: Path,
    top_color: tuple[int, int, int],
    bottom_color: tuple[int, int, int],
    title: str,
    subtitle: str,
    shape_type: str = "rect",
) -> Path:
    """Create a stylized illustration-like image using PIL."""
    width, height = 512, 512
    img = Image.new("RGB", (width, height), top_color)
    draw = ImageDraw.Draw(img)

    # Vertical gradient
    for y in range(height):
        ratio = y / height
        r = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
        g = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
        b = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Decorative shapes based on category
    if shape_type == "car":
        # Road
        draw.rectangle([0, 360, width, height], fill=(40, 44, 52))
        draw.line([0, 430, width, 430], fill=(255, 215, 0), width=6)
        # Car body (red sports car)
        draw.polygon([(110, 370), (160, 300), (340, 300), (400, 370)], fill=(220, 20, 60))
        draw.rectangle([90, 350, 420, 390], fill=(200, 15, 45))
        # Cabin / window
        draw.polygon([(170, 305), (200, 330), (310, 330), (330, 305)], fill=(173, 216, 230))
        # Wheels
        draw.ellipse([140, 370, 190, 420], fill=(20, 20, 20), outline=(180, 180, 180), width=4)
        draw.ellipse([320, 370, 370, 420], fill=(20, 20, 20), outline=(180, 180, 180), width=4)
    elif shape_type == "dog":
        # Green lawn
        draw.rectangle([0, 340, width, height], fill=(46, 139, 87))
        # Golden puppy silhouette
        draw.ellipse([170, 260, 340, 370], fill=(218, 165, 32))  # body
        draw.ellipse([270, 200, 370, 290], fill=(218, 165, 32))  # head
        draw.polygon([(340, 220), (370, 260), (330, 260)], fill=(184, 134, 11))  # ear
        draw.ellipse([345, 235, 360, 250], fill=(10, 10, 10))  # eye
        draw.ellipse([360, 250, 375, 265], fill=(10, 10, 10))  # nose
        draw.rectangle([180, 350, 210, 400], fill=(184, 134, 11))  # paw 1
        draw.rectangle([290, 350, 320, 400], fill=(184, 134, 11))  # paw 2
    elif shape_type == "fruit":
        # Table
        draw.rectangle([0, 350, width, height], fill=(139, 69, 19))
        # Bowl
        draw.chord(
            [120, 260, 390, 420], 0, 180, fill=(245, 245, 245), outline=(200, 200, 200), width=3
        )
        # Apple (Red)
        draw.ellipse([170, 250, 250, 330], fill=(220, 20, 60))
        # Orange
        draw.ellipse([250, 260, 320, 330], fill=(255, 140, 0))
        # Banana arc
        draw.arc([190, 220, 340, 340], 30, 150, fill=(255, 215, 0), width=18)
    elif shape_type == "laptop":
        # Desk
        draw.rectangle([0, 380, width, height], fill=(160, 110, 70))
        # Laptop screen
        draw.rectangle([140, 180, 370, 340], fill=(30, 30, 30), outline=(180, 180, 180), width=4)
        # Code lines in screen
        for line_y, fill_col in [
            (210, (100, 200, 255)),
            (230, (255, 180, 100)),
            (250, (120, 255, 150)),
            (270, (220, 120, 255)),
            (290, (100, 200, 255)),
        ]:
            draw.rectangle([160, line_y, 320, line_y + 8], fill=fill_col)
        # Keyboard base
        draw.polygon([(110, 375), (140, 340), (370, 340), (400, 375)], fill=(120, 125, 130))
    elif shape_type == "piano":
        # Stage background
        draw.rectangle([0, 200, width, height], fill=(20, 20, 25))
        # Piano body
        draw.rectangle([60, 260, 450, 420], fill=(10, 10, 10), outline=(218, 165, 32), width=3)
        # White keys
        key_width = 24
        for k in range(15):
            kx = 75 + k * key_width
            draw.rectangle(
                [kx, 300, kx + key_width - 2, 400], fill=(250, 250, 250), outline=(50, 50, 50)
            )
        # Black keys
        for k in [0, 1, 3, 4, 5, 7, 8, 10, 11, 12]:
            bx = 75 + k * key_width + 16
            draw.rectangle([bx, 300, bx + 14, 360], fill=(15, 15, 15))
    elif shape_type == "rain":
        # Stormy clouds
        draw.ellipse([80, 80, 260, 200], fill=(70, 80, 95))
        draw.ellipse([200, 60, 380, 210], fill=(80, 90, 105))
        draw.ellipse([300, 90, 440, 200], fill=(60, 70, 85))
        # Rain streaks
        random.seed(42)
        for _ in range(80):
            rx = random.randint(50, 460)
            ry = random.randint(220, 480)
            draw.line([(rx, ry), (rx - 5, ry + 25)], fill=(173, 216, 230), width=2)

    # Title Banner
    draw.rectangle([0, 0, width, 55], fill=(0, 0, 0, 160))
    draw.text((20, 12), f"{title}", fill=(255, 255, 255))
    draw.text((20, 32), f"{subtitle}", fill=(200, 200, 200))

    img.save(filename)
    return filename


def generate_synthetic_wav(
    filename: Path,
    sound_type: str = "piano",
    duration_seconds: float = 3.0,
    sample_rate: int = 22050,
) -> Path:
    """Generate synthetic WAV audio without external heavy audio libraries."""
    num_samples = int(duration_seconds * sample_rate)
    samples: list[int] = []

    if sound_type == "piano":
        # Play a sequence of harmonic piano-like notes: A4 (440Hz), C#5 (554Hz), E5 (659Hz)
        frequencies = [440.0, 554.37, 659.25, 880.0]
        note_duration = num_samples // len(frequencies)

        for i in range(num_samples):
            note_idx = min(i // note_duration, len(frequencies) - 1)
            freq = frequencies[note_idx]
            t = (i % note_duration) / sample_rate

            # Exponential decay envelope (like a plucked string / struck piano hammer)
            decay = math.exp(-3.5 * t)
            # Fundamental + 1st harmonic
            val = (
                math.sin(2 * math.pi * freq * t) + 0.5 * math.sin(2 * math.pi * 2 * freq * t)
            ) * decay
            clamped = max(-1.0, min(1.0, val))
            samples.append(int(clamped * 32767 * 0.7))
    else:  # Rain sound
        # Modulated noise
        random.seed(1337)
        for i in range(num_samples):
            t = i / sample_rate
            # Low frequency swell (storm wave)
            swell = 0.5 + 0.5 * math.sin(2 * math.pi * 0.5 * t)
            white_noise = (random.random() * 2.0 - 1.0) * swell
            clamped = max(-1.0, min(1.0, white_noise))
            samples.append(int(clamped * 32767 * 0.35))

    with wave.open(str(filename), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        raw_data = struct.pack(f"<{len(samples)}h", *samples)
        wav_file.writeframes(raw_data)

    return filename


def generate_sample_media(output_dir: str | Path | None = None) -> dict[str, list[Path]]:
    """Generate all media files recommended by the DataCamp tutorial."""
    target_dir = Path(output_dir or settings.get_media_path())
    target_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, list[Path]] = {"images": [], "audio": [], "queries": []}

    # 1. Image collection
    images_metadata = [
        (
            "car.png",
            (30, 40, 60),
            (90, 110, 140),
            "Red Sports Car",
            "A sports car parked on a paved street",
            "car",
        ),
        (
            "dog.png",
            (135, 206, 235),
            (255, 255, 255),
            "Golden Retriever",
            "A playful dog running on green grass",
            "dog",
        ),
        (
            "fruit.png",
            (240, 230, 220),
            (200, 180, 160),
            "Fresh Fruit Bowl",
            "Apples, bananas, and oranges on a table",
            "fruit",
        ),
        (
            "laptop.png",
            (40, 45, 55),
            (20, 25, 30),
            "Developer Laptop",
            "A laptop workspace with software code",
            "laptop",
        ),
        (
            "piano.png",
            (40, 30, 50),
            (15, 10, 20),
            "Grand Piano",
            "A grand piano keyboard with black and white keys",
            "piano",
        ),
        (
            "rain.png",
            (40, 50, 65),
            (70, 85, 100),
            "Gentle Rain",
            "Dark storm clouds with rain falling",
            "rain",
        ),
    ]

    for fname, top_col, bot_col, title, subtitle, shape in images_metadata:
        path = create_gradient_image(target_dir / fname, top_col, bot_col, title, subtitle, shape)
        generated["images"].append(path)

    # 2. Query Image (used for Step 6: Search images with an image)
    query_image_path = (
        target_dir.parent / "query.jpg" if target_dir.name == "media" else target_dir / "query.jpg"
    )
    create_gradient_image(
        query_image_path,
        (20, 30, 50),
        (80, 90, 120),
        "Query Image: Red Automobile",
        "A red vehicle used as a visual query",
        "car",
    )
    generated["queries"].append(query_image_path)

    # 3. Audio clips (Step 7: Search audio with text)
    piano_wav = generate_synthetic_wav(target_dir / "piano_melody.wav", sound_type="piano")
    rain_wav = generate_synthetic_wav(target_dir / "rain_storm.wav", sound_type="rain")
    generated["audio"].extend([piano_wav, rain_wav])

    return generated
