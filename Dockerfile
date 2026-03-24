FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 7860

# 启动两个服务（简化版，生产环境建议用docker-compose）
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port 8000 & python app.py"]