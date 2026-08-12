import multiprocessing
import os

# Server socket binding
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")

# Worker processes configuration
# Formula for standard web apps: (2 * cores) + 1
# IMPORTANT FOR RAG: Sentence-Transformers / CrossEncoders load model weights in each worker process (~200MB each).
# - For t2.micro (1 vCPU, 1GB RAM) -> default to 1 worker to stay within 1GB RAM
# - For larger instances (2+ vCPUs, 4GB+ RAM) -> set WEB_CONCURRENCY=2 or 4 via environment
workers_per_core_str = os.getenv("WORKERS_PER_CORE", "1")
web_concurrency = os.getenv("WEB_CONCURRENCY", None)

if web_concurrency:
    workers = int(web_concurrency)
else:
    cores = multiprocessing.cpu_count()
    workers = max(int(float(workers_per_core_str) * cores), 1)

# Use Uvicorn ASGI Worker
worker_class = "uvicorn.workers.UvicornWorker"

# Timeouts
# LLM generation and PDF parsing can take time — avoid premature 504 gateway timeouts
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))

# Logging
loglevel = os.getenv("LOG_LEVEL", "info")
accesslog = "-"  # Log access events to stdout
errorlog = "-"   # Log errors to stderr
