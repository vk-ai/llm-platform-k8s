FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[qdrant,pgvector]" && useradd -u 10001 -m app
COPY config ./config
USER 10001
EXPOSE 8080
ENTRYPOINT ["llm-platform"]
CMD ["serve", "--port", "8080"]
