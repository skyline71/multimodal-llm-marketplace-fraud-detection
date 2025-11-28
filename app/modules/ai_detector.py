from transformers import pipeline
from PIL import Image

class AIDetector:
    def __init__(self):
        # Инициализируем пайплайн классификации изображений.
        # Модель автоматически скачается с Hugging Face (около 100-200 МБ)
        # При первом запуске это займет время!
        print("Загрузка модели AI Detector...")
        self.pipe = pipeline("image-classification", model="umm-maybe/AI-image-detector")

    def detect_ai_image(self, image: Image.Image) -> dict:
        """
        Определяет, сгенерировано ли изображение ИИ, используя нейросеть.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Прогоняем изображение через модель
        results = self.pipe(image)
        
        # Результат приходит в формате списка словарей: [{'label': 'artificial', 'score': 0.99}, ...]
        # Нам нужно найти score для метки 'artificial' (или 'fake')
        
        ai_score = 0.0
        is_ai = False
        
        # Разбираем результаты (обычно там метки 'artificial' и 'real')
        for res in results:
            if res['label'].lower() in ['artificial', 'fake']:
                ai_score = res['score']
        
        # Порог срабатывания (например, 60%)
        if ai_score > 0.6:
            is_ai = True
            explanation = f"Нейросеть обнаружила признаки генерации (уверенность: {ai_score:.2%})"
        else:
            explanation = "Изображение классифицировано как естественное."

        return {
            "is_ai_generated": is_ai,
            "ai_score": round(ai_score, 2),
            "explanation": explanation
        }