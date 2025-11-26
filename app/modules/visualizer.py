# app/modules/visualizer.py

import streamlit as st
from PIL import ImageDraw

def draw_bounding_boxes(image, results):
    """Добавляет bounding boxes к изображению."""
    draw = ImageDraw.Draw(image)
    boxes = results[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id]
        draw.text((x1, y1), label, fill="red")
    return image

def render_report(analysis_result, original_image, yolo_results):
    """Отображает отчёт в Streamlit."""
    st.subheader("Результат анализа")
    st.write(f"**Уровень риска:** {analysis_result['risk_level']}")
    st.write(f"**Сходство изображение-текст:** {analysis_result['similarity_score']}")
    st.write(f"**Обнаруженные объекты:** {', '.join(analysis_result['detected_objects'])}")

    annotated_img = draw_bounding_boxes(original_image.copy(), yolo_results)
    st.image(annotated_img, caption="Анализ изображения", use_container_width=True)
