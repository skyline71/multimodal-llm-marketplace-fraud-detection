import streamlit as st
from PIL import Image
from modules.analyzer import LotAnalyzer
from modules.rag_llm import RAGLLM
from modules.visualizer import draw_bounding_boxes
import os

# Кэшируем загрузку моделей, чтобы не перезагружать их при каждом клике
@st.cache_resource
def load_models():
    return LotAnalyzer(), RAGLLM()

analyzer, rag_llm = load_models()

st.set_page_config(page_title="Анализ мошенничества", layout="wide", page_icon="🛡️")
st.title("Аналитическая система выявления признаков мошенничества в лотах маркетплейсов")
st.markdown("Проверка на соответствие описания, детекция ИИ-фейков и поиск аналогов.")

# === Боковая панель для выбора режима ===
with st.sidebar:
    st.header("Настройки")
    mode = st.radio(
        "Источник данных:",
        ["📤 Загрузка файла (Ручной режим)", "🌐 По ссылке (Маркетплейс)"],
        index=0
    )
    st.info("Система использует YOLOv8 для детекции, CLIP для семантики и ResNet для поиска дипфейков.")

# === Функция запуска анализа (общая логика) ===
def run_full_analysis(image, text_input, lot_id_prefix):
    if not image or not text_input:
        st.error("Необходимо загрузить изображение и ввести описание!")
        return

    # Генерируем ID
    lot_id = f"{lot_id_prefix}_{str(hash(text_input))[-6:]}"

    try:
        with st.spinner("⏳ Выполняется комплексный анализ..."):
            # 1. Основной анализ
            analysis = analyzer.analyze_lot(image, text_input, lot_id)

            # --- Блок 1: Визуализация ---
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📸 Анализ изображения")
                annotated_img = draw_bounding_boxes(image.copy(), analysis["yolo_results"])
                st.image(annotated_img, caption="Детекция объектов (YOLOv8)", use_container_width=True)
                st.write(f"**Найденные объекты:** {', '.join(analysis['detected_objects'])}")

            with col2:
                st.subheader("🧠 Метрики системы")
                
                # ИИ Детектор
                ai_data = analysis["ai_detection"]
                if ai_data["is_ai_generated"]:
                    st.error(f"⚠️ **ИИ-Генерация:** {ai_data['ai_score']*100:.1f}%")
                else:
                    st.success(f"✅ **Изображение:** Реальное (Score: {ai_data['ai_score']:.2f})")
                st.caption(ai_data["explanation"])

                st.divider()

                # Сходство текста
                sim_score = analysis["similarity_score"]
                if sim_score < 0.3:
                    st.error(f"📉 **Сходство текст-фото:** Низкое ({sim_score:.2f})")
                elif sim_score < 0.5:
                    st.warning(f"⚠️ **Сходство текст-фото:** Среднее ({sim_score:.2f})")
                else:
                    st.success(f"📈 **Сходство текст-фото:** Высокое ({sim_score:.2f})")

            st.divider()

            # --- Блок 2: LLM Отчет и RAG ---
            col_rag, col_llm = st.columns([1, 2])

            with col_rag:
                st.subheader("📚 База знаний (RAG)")
                if analysis["rag_context"]:
                    for case in analysis["rag_context"]:
                        with st.expander(f"Похожий кейс ({case['risk_level']})"):
                            st.write(case['description'])
                            st.caption(f"Совет: {case['recommendation']}")
                else:
                    st.info("Похожих подозрительных случаев не найдено.")

            with col_llm:
                st.subheader("📝 Вердикт AI-ассистента")
                # Генерация отчёта LLM
                report_placeholder = st.empty()
                report_placeholder.text("Анализируем данные...")
                
                # ИСПРАВЛЕНИЕ: Передаем text_input вторым аргументом!
                report = rag_llm.generate_report(analysis, text_input)
                
                report_placeholder.markdown(report)

            # Итоговый статус
            st.divider()
            if analysis["risk_level"] == "высокий":
                st.error(f"🛑 ИТОГОВЫЙ РИСК: ВЫСОКИЙ. Лот выглядит подозрительно.")
            elif analysis["risk_level"] == "средний":
                st.warning(f"⚠️ ИТОГОВЫЙ РИСК: СРЕДНИЙ. Требуется проверка.")
            else:
                st.success(f"✅ ИТОГОВЫЙ РИСК: НИЗКИЙ. Лот безопасен.")

    except Exception as e:
        st.error(f"Произошла ошибка во время анализа: {e}")
        st.exception(e)


# === РЕЖИМ 1: Загрузка файла ===
if mode == "📤 Загрузка файла (Ручной режим)":
    st.subheader("Ручной анализ данных")
    
    col_in1, col_in2 = st.columns([1, 1])
    
    with col_in1:
        uploaded_file = st.file_uploader("1. Загрузите фото товара", type=["jpg", "jpeg", "png", "webp"])
    
    with col_in2:
        user_text = st.text_area("2. Введите название и описание товара", height=100, placeholder="Например: Смартфон Apple iPhone 13 128GB...")

    start_btn = st.button("🚀 Запустить анализ", type="primary", use_container_width=True)

    if start_btn:
        if uploaded_file and user_text:
            image = Image.open(uploaded_file).convert("RGB")
            run_full_analysis(image, user_text, "manual_upload")
        else:
            st.warning("Пожалуйста, загрузите изображение и введите текст описания.")

# === РЕЖИМ 2: По ссылке ===
elif mode == "🌐 По ссылке (Маркетплейс)":
    st.subheader("Автоматический анализ по ссылке")
    st.text_input("Ссылка на товар (Ozon / WB)", disabled=True, placeholder="https://...")
    
    st.warning("""
    ⚠️ **Функционал временно недоступен**
    
    Прямой парсинг маркетплейсов временно отключен в связи с обновлением защиты от ботов на стороне Ozon и Wildberries (CAPTCHA / Cloudflare).
    
    Пожалуйста, используйте режим **"Загрузка файла"**:
    1. Сохраните фото товара вручную.
    2. Скопируйте описание.
    3. Загрузите в соседней вкладке.
    """)