# Builds this Containers checkout. Never FROM ghcr.io.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . "datasets>=2.14"

EXPOSE 8080
CMD ["python", "-m", "synth_containers.local_serve"]
