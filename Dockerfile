FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements.txt

COPY . /app

RUN python scripts/smoke_test.py

ENTRYPOINT ["python"]
CMD ["src/train.py", "--help"]
