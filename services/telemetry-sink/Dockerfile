FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt influxdb-client

COPY shared/ ./shared/
COPY services/telemetry-sink/ ./services/telemetry-sink/

ENV PYTHONPATH=/app/shared:/app/services:/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "services/telemetry-sink/main.py"]
