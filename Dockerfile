FROM mcr.microsoft.com/playwright/python:v1.49.0-jammy
RUN apt-get update && apt-get install -y fonts-noto-core fonts-noto-color-emoji fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY bot.py .
CMD ["python", "bot.py"]