FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    numpy \
    "openenv-core==0.2.1" \
    "transformers>=4.40.0" \
    "torch>=2.2.0" \
    "accelerate>=0.30.0" \
    "sentencepiece>=0.2.0" \
    "safetensors>=0.4.0"

# Copy source
COPY server/ server/
COPY viz_standalone.html .

# HF Spaces runs on port 7860
ENV PORT=7860
ENV DIFFICULTY=easy

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
