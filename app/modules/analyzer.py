# app/modules/analyzer.py
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from ultralytics import YOLO
from .ai_detector import AIDetector

class LotAnalyzer:
    def __init__(self):
        self.yolo = YOLO("yolov8n.pt")
        self.clip = SentenceTransformer('clip-ViT-B-32')
        self.ai_detector = AIDetector()
        # База знаний для RAG (будет обновляться)
        self.knowledge_base = [
            {
                "description": "Лот с изображением человека в категории 'офисный стул'",
                "risk_level": "высокий",
                "recommendation": "Не рекомендуется к покупке"
            },
            {
                "description": "Несоответствие текста и изображения: текст про 'наушники', на фото - человек",
                "risk_level": "средний",
                "recommendation": "Проверьте отзывы продавца"
            }
        ]

    def analyze_lot(self, image: Image.Image, text: str, lot_id: str = "demo"):
        # 1. Детекция объектов
        yolo_results = self.yolo(image)
        detected_objects = [yolo_results[0].names[int(cls)] for cls in yolo_results[0].boxes.cls]
        
        # 2. Детекция ИИ-изображения
        ai_result = self.ai_detector.detect_ai_image(image)
        
        # 3. Семантическое сходство
        img_emb = self.clip.encode(image, convert_to_tensor=True)
        text_emb = self.clip.encode(text, convert_to_tensor=True)
        similarity = util.cos_sim(img_emb, text_emb).item()
        
        # 4. Логические правила
        suspicious_objects = ["person", "car", "animal"]  # Объекты, неожиданные для многих категорий
        has_suspicious = any(obj in detected_objects for obj in suspicious_objects)
        
        # 5. RAG-поиск похожих случаев
        rag_results = self._search_similar_cases(text, detected_objects, similarity)
        
        # 6. Формирование вердикта
        risk_level = "низкий"
        if ai_result["is_ai_generated"] or similarity < 0.2 or has_suspicious:
            risk_level = "высокий"
        elif similarity < 0.4:
            risk_level = "средний"
            
        return {
            "lot_id": lot_id,
            "detected_objects": detected_objects,
            "similarity_score": round(similarity, 3),
            "ai_detection": ai_result,
            "risk_level": risk_level,
            "rag_context": rag_results,
            "yolo_results": yolo_results  # Для визуализации
        }
    
    def _search_similar_cases(self, text: str, objects: list, similarity: float):
        """Поиск похожих случаев в базе знаний (заглушка для ChromaDB)"""
        results = []
        for case in self.knowledge_base:
            # Простой поиск по ключевым словам
            if any(obj in case["description"].lower() for obj in objects) or \
               any(word in text.lower() for word in ["стул", "наушники", "телефон"]):
                results.append(case)
        return results[:2]  # Возвращаем максимум 2 похожих случая