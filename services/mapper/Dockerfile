FROM python:3.12-slim

WORKDIR /app

# Dependencies first, so a code change does not invalidate the pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY services/localization/ ./services/localization/
COPY services/mapping/ ./services/mapping/
COPY services/mapper/ ./services/mapper/
COPY simulator/ ./simulator/

ENV PYTHONPATH=/app/shared:/app/services:/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=3)"

CMD ["python", "services/mapper/main.py", "--source", "mqtt", "--port", "8080"]
