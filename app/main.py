# app/main.py
import streamlit as st
from PIL import Image
from modules.analyzer import LotAnalyzer
from modules.rag_llm import RAGLLM
from modules.visualizer import draw_bounding_boxes
import os

# Инициализация компонентов
analyzer = LotAnalyzer()
rag_llm = RAGLLM()

st.set_page_config(page_title="Анализ мошенничества", layout="wide")
st.title("🔍 Аналитическая система: выявление подозрительных лотов")

# Загрузка данных
data_dir = "data"
lot_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
selected_lot = st.selectbox("Выберите лот для анализа", lot_dirs)

if selected_lot and st.button("🚀 Запустить анализ"):
    lot_path = os.path.join(data_dir, selected_lot)
    
    try:
        # Загрузка изображения и текста
        image = Image.open(os.path.join(lot_path, "image.jpg")).convert("RGB")
        with open(os.path.join(lot_path, "description.txt"), "r", encoding="utf-8") as f:
            text = f.read().strip()
        
        # Отображение исходных данных
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Исходное изображение", use_container_width=True)
        with col2:
            st.subheader("Описание товара:")
            st.text_area("", text, height=150, disabled=True)
        
        # Анализ
        with st.spinner("Анализируем лот..."):
            analysis = analyzer.analyze_lot(image, text, selected_lot)
            
            # Визуализация результатов YOLO
            annotated_img = draw_bounding_boxes(image.copy(), analysis["yolo_results"])
            st.subheader("📊 Результаты анализа изображения:")
            st.image(annotated_img, caption="Обнаруженные объекты", use_container_width=True)
            
            # Детектор ИИ
            st.subheader("🤖 Анализ ИИ-генерации:")
            ai_status = "⚠️ **Подозрение на ИИ-генерацию**" if analysis["ai_detection"]["is_ai_generated"] else "✅ Изображение естественное"
            st.markdown(f"{ai_status} (Вероятность: {analysis['ai_detection']['ai_score']:.2f})")
            st.caption(analysis["ai_detection"]["explanation"])
            
            # Сходство текста и изображения
            st.subheader("🔄 Сходство текста и изображения:")
            similarity_color = "red" if analysis["similarity_score"] < 0.3 else "orange" if analysis["similarity_score"] < 0.5 else "green"
            st.markdown(f"<h3 style='color:{similarity_color}'>Оценка сходства: {analysis['similarity_score']:.2f}</h3>", 
                       unsafe_allow_html=True)
            
            # RAG-контекст
            if analysis["rag_context"]:
                st.subheader("📚 Похожие случаи из базы знаний:")
                for case in analysis["rag_context"]:
                    st.info(f"**{case['risk_level'].capitalize()} риск:** {case['description']}")
                    st.caption(f"Рекомендация: {case['recommendation']}")
            
            # Генерация отчёта LLM
            st.subheader("📝 Генерация отчёта (LLM):")
            with st.spinner("Генерируем детальный отчёт..."):
                report = rag_llm.generate_report(analysis)
                st.markdown(report)
            
            # Итоговый вердикт
            st.subheader("🎯 Итоговый вердикт:")
            risk_colors = {"низкий": "green", "средний": "orange", "высокий": "red"}
            st.markdown(f"<h2 style='color:{risk_colors[analysis['risk_level']]}'>Уровень риска: {analysis['risk_level'].upper()}</h2>", 
                       unsafe_allow_html=True)
            
            if analysis["risk_level"] == "высокий":
                st.error("⚠️ **Рекомендация: Не рекомендуется к покупке**")
            elif analysis["risk_level"] == "средний":
                st.warning("⚠️ **Рекомендация: Проверьте отзывы и рейтинг продавца**")
            else:
                st.success("✅ **Рекомендация: Лот выглядит безопасным для покупки**")
                
    except Exception as e:
        st.error(f"Ошибка при анализе: {str(e)}")
        st.exception(e)