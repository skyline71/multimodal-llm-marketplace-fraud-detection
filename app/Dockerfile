# Dockerfile (минималистичный, рабочий на RTX 3050)
FROM python:3.10-slim

# Установка только необходимых системных пакетов
RUN apt-get update && apt-get install -y \
    chromium-driver \
    chromium \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Установка Python-зависимостей
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование проекта
COPY . .

# Запуск Streamlit
CMD ["python3", "-m", "streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]