# app/modules/ai_detector.py

from PIL import Image
import numpy as np

class AIDetector:
    def detect_ai_image(self, image: Image.Image) -> dict:
        """
        Эвристический детектор ИИ-изображений.
        Основан на типичных признаках:
        - Очень высокое разрешение
        - Низкий уровень шума (гладкость)
        - Отсутствие JPEG-артефактов
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        img_array = np.array(image)
        
        # Признак 1: высокое разрешение (>1024x1024) — частый признак ИИ
        is_high_res = image.width >= 1024 and image.height >= 1024
        
        # Признак 2: низкое стандартное отклонение = гладкое изображение
        std_dev = img_array.std()
        is_smooth = std_dev < 35  # Эмпирический порог
        
        # Признак 3: отсутствие JPEG-артефактов (часто в ИИ)
        # Упрощённо: если формат не JPEG → подозрительно
        # (но в нашем случае изображения уже в PIL, поэтому пропустим)
        
        # Оценка
        ai_score = 0.1
        if is_high_res and is_smooth:
            ai_score = 0.9
        elif is_high_res or is_smooth:
            ai_score = 0.6
        
        return {
            "is_ai_generated": ai_score > 0.7,
            "ai_score": round(ai_score, 2),
            "explanation": (
                "Высокое разрешение и гладкость изображения указывают на возможную генерацию ИИ"
                if ai_score > 0.7 else
                "Изображение выглядит как обычная фотография"
            )
        }