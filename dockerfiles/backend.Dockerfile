
FROM python:3.12-slim

WORKDIR /app

COPY requirements.in requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY config/ /app/config/

COPY src/backend/ /app/src/backend/
COPY src/__init__.py /app/src/__init__.py

EXPOSE 8000

ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["python", "-m", "src.backend.api.main"]