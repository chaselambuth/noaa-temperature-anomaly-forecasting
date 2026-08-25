FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY app.py ./app.py
COPY modeling.py ./modeling.py
COPY static ./static
COPY templates ./templates
COPY README.md ./README.md
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

EXPOSE 5000
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["python", "-c", "from app import app; app.run(host='0.0.0.0', port=5000)"]
