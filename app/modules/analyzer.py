# app/modules/analyzer.py

from PIL import Image
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer, util
import chromadb
from .ai_detector import AIDetector
from .vector_db import VectorDB
import os

class LotAnalyzer:
    def __init__(self):
        # Модели компьютерного зрения
        self.yolo = YOLO("yolov8n.pt")
        self.clip = SentenceTransformer('clip-ViT-B-32')
        self.ai_detector = AIDetector()
        self.vector_db = VectorDB()

    def get_category_from_text(self, text: str) -> str:
        """Определяет категорию по ключевым словам (можно расширить)"""
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["стул", "кресло", "стол", "мебель"]):
            return "мебель"
        elif any(kw in text_lower for kw in ["наушники", "колонка", "плеер", "аудио"]):
            return "аудиотехника"
        elif any(kw in text_lower for kw in ["телефон", "смартфон", "айфон", "самсунг"]):
            return "телефоны"
        elif any(kw in text_lower for kw in ["шлепанцы", "обувь", "ботинки", "кроссовки"]):
            return "обувь"
        else:
            return "другое"

    def analyze_lot(self, image: Image.Image, text: str, lot_id: str = "demo"):
        # 1. Детекция объектов
        yolo_results = self.yolo(image)
        detected_objects = [yolo_results[0].names[int(cls)] for cls in yolo_results[0].boxes.cls]

        # 2. Детекция ИИ-генерации
        ai_result = self.ai_detector.detect_ai_image(image)

        # 3. Семантическое сходство
        img_emb = self.clip.encode(image, convert_to_tensor=True)
        text_emb = self.clip.encode(text, convert_to_tensor=True)
        similarity = util.cos_sim(img_emb, text_emb).item()

        # 4. Логические правила по категории
        category = self.get_category_from_text(text)
        forbidden_objects = {
            "мебель": ["person", "car", "animal", "laptop"],
            "аудиотехника": ["person", "car", "animal", "food"],
            "телефоны": ["person", "car", "animal", "cat", "dog"],
            "обувь": ["person", "car", "laptop", "book"],
        }
        forbidden_list = forbidden_objects.get(category, [])
        has_forbidden = any(obj in detected_objects for obj in forbidden_list)

        # 5. Определение уровня риска
        risk_level = "низкий"
        if ai_result["is_ai_generated"] or similarity < 0.2 or has_forbidden:
            risk_level = "высокий"
        elif similarity < 0.4:
            risk_level = "средний"

        # 6. Сохранение лота в ChromaDB для будущих RAG-поисков
        verdict_summary = (
            "Подозрительный лот: высокое несоответствие или ИИ-изображение"
            if risk_level == "высокий"
            else "Несоответствие текста и изображения"
        )
        self.vector_db.add_lot(
            lot_id=lot_id,
            text=text,
            detected_objects=detected_objects,
            risk_level=risk_level,
            verdict=verdict_summary
        )

        # 7. RAG: поиск похожих случаев
        query_text = f"Товар: {text}. Категория: {category}. Объекты: {', '.join(detected_objects)}"
        rag_context = self.vector_db.query_similar(query_text, top_k=2)

        return {
            "lot_id": lot_id,
            "detected_objects": detected_objects,
            "similarity_score": round(similarity, 3),
            "ai_detection": ai_result,
            "risk_level": risk_level,
            "rag_context": rag_context,
            "yolo_results": yolo_results,
            "category": category,
            "has_forbidden": has_forbidden,
            "forbidden_objects": forbidden_list
        }