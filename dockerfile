FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
  && rm -rf /var/lib/apt/lists/*

COPY requirements_docker.txt /app/requirements_docker.txt
RUN pip install --no-cache-dir -r /app/requirements_docker.txt

# код
COPY src/ /app/src/

COPY artifacts/medsearch-miniLM/ /app/artifacts/medsearch-miniLM/
# видимость пакета
ENV PYTHONPATH=/app/src 

ENTRYPOINT ["python", "-m", "med_entity_clf.cli_predict"]