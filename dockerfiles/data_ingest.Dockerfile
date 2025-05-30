FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY config/ /app/config/

COPY src/backend/ /app/src/backend/
COPY src/__init__.py /app/src/__init__.py

EXPOSE 8080

ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["uvicorn", "src.backend.api.main_data_ingest:app", "--host", "0.0.0.0", "--port", "8080"]
