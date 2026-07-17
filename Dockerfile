FROM python:3.11-slim

# build-essential: insightface builds from source
# libgl1 + libglib2.0-0: insightface pulls in the full opencv-python (needs libGL)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model into the image so startup doesn't have to download ~300MB
ARG MODEL_NAME=buffalo_l
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='${MODEL_NAME}', providers=['CPUExecutionProvider']).prepare(ctx_id=-1)"

COPY app ./app

ENV DATA_DIR=/srv/data
EXPOSE 8000

# Use exactly 1 worker: the matcher keeps its state in process RAM
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
