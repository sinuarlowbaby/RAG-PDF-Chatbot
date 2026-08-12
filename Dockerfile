# Use Python 3.11 — required for full langfuse v2.x compatibility
# (langfuse.decorators emits SyntaxWarnings on Python 3.14+)
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/app:/app"

# Set the working directory in the container
WORKDIR /app

# Install uv directly from PyPI (fast global CDN, avoids slow ghcr.io downloads)
RUN pip install --no-cache-dir uv

# Install lightweight CPU-only PyTorch using uv (~150MB instead of 2.5GB CUDA)
RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies before copying source (better layer caching)
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Pre-download the HuggingFace CrossEncoder model so startup is instant and offline-ready
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy the project files
COPY . .

# Create a non-root user and switch to it (security best practice)
RUN adduser --disabled-password --gecos "" appuser \
    && mkdir -p /app/app/uploads /app/.cache \
    && chown -R appuser:appuser /app
USER appuser

# Expose the port
EXPOSE 8000

# Set the execution directory to the app folder
WORKDIR /app/app

# Run the application with Gunicorn managing Uvicorn workers
CMD ["gunicorn", "-c", "/app/gunicorn_conf.py", "app:app"]