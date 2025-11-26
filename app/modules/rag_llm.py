# app/modules/rag_llm.py
import requests
import json

class RAGLLM:
    def __init__(self):
        self.ollama_url = "http://ollama:11434/api/generate"
        self.model = "qwen:4b"  # 4-bit квантованная версия для RTX 3050
        
    def generate_report(self, analysis_result: dict) -> str:
        """Генерирует текстовый отчет с помощью LLM"""
        # Формируем контекст для LLM
        context = f"""
Анализ лота #{analysis_result['lot_id']}:
- Обнаруженные объекты: {', '.join(analysis_result['detected_objects'])}
- Сходство изображения и текста: {analysis_result['similarity_score']:.2f}
- Риск ИИ-генерации: {'Да' if analysis_result['ai_detection']['is_ai_generated'] else 'Нет'} (вероятность: {analysis_result['ai_detection']['ai_score']:.2f})
- Уровень риска: {analysis_result['risk_level'].upper()}
        
Похожие случаи из базы знаний:
"""
        for i, case in enumerate(analysis_result['rag_context'], 1):
            context += f"{i}. {case['description']} - Риск: {case['risk_level']}\n"
        
        # Формируем запрос к LLM
        prompt = f"""
Ты — эксперт по анализу мошенничества на маркетплейсах. На основе следующих данных сформируй структурированный отчет для покупателя:

{context}

Отчет должен содержать:
1. Краткое заключение (1-2 предложения)
2. Конкретные признаки риска (если есть)
3. Рекомендации покупателю
4. Ссылки на похожие случаи (если применимо)

Пиши на русском языке, используй формальный стиль, но понятный для обычного пользователя.
"""
        
        try:
            # Запрос к Ollama
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Ошибка генерации отчёта: {response.text}"
                
        except Exception as e:
            return f"Не удалось подключиться к LLM: {str(e)}. Используйте локальный режим анализа."
