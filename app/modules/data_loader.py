# app/modules/data_loader.py

import os
from PIL import Image

def load_local_data(image_path: str, text_path: str):
    """Загружает изображение и текст из локальных файлов."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Text not found: {text_path}")

    image = Image.open(image_path).convert("RGB")
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return image, text


def mock_input():
    """Заглушка для демо: возвращает путь к тестовым данным."""
    return "data/demo_image.jpg", "data/demo_text.txt"
