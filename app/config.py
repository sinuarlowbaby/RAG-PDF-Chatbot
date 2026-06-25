"""
app/config.py
─────────────────────────────────────────────────────────────────────────────
Centralised application settings loaded from environment variables / .env file.

Usage anywhere in the app:
    from config import settings
    url = settings.qdrant_url
─────────────────────────────────────────────────────────────────────────────
"""
import logging
import os
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Resolve paths relative to this file, not the CWD ────────────────────────
_APP_DIR = Path(__file__).parent
_ROOT_DIR = _APP_DIR.parent


class Settings(BaseSettings):
    """All application configuration in one place.

    Values are read (in priority order) from:
      1. Real environment variables
      2. The .env file at the project root
      3. The defaults defined below
    """

    model_config = SettingsConfigDict(
        env_file=str(_ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",        # silently ignore unknown env vars
    )

    # ── LLM Providers ────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    groq_api_key: str = Field(default="", description="Groq API key (optional)")

    # ── LangSmith Tracing ────────────────────────────────────────────────────
    langsmith_tracing: bool = Field(default=False)
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com")
    langsmith_api_key: str = Field(default="")
    langsmith_project: str = Field(default="RAG-PDF-Chatbot")

    # ── Infrastructure ───────────────────────────────────────────────────────
    qdrant_url: str = Field(default="http://localhost:6333")
    redis_url: str = Field(default="redis://localhost:6379")
    qdrant_collection_name: str = Field(default="global_rag_store")

    # ── CORS ─────────────────────────────────────────────────────────────────
    # Stored as a raw comma-separated string in env; parsed into a list below.
    allowed_origins: str = Field(default="http://localhost:8000")

    # ── Embedding model ──────────────────────────────────────────────────────
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536)
    embedding_chunk_size: int = Field(default=100)

    # ── Reranker model ───────────────────────────────────────────────────────
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ── Storage ──────────────────────────────────────────────────────────────
    upload_dir: Path = Field(default=_APP_DIR / "uploads")
    max_upload_size_mb: int = Field(default=50)

    # ── Server ───────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=True)
    log_level: str = Field(default="info")

    # ── Logging ──────────────────────────────────────────────────────────────
    log_file: Path = Field(default=_APP_DIR / "app.log")

    # ── Computed properties ──────────────────────────────────────────────────
    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse comma-separated ALLOWED_ORIGINS into a Python list."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def redis_host(self) -> str:
        """Extract hostname from redis_url (e.g. 'redis://localhost:6379')."""
        return self.redis_url.replace("redis://", "").split(":")[0]

    @property
    def redis_port(self) -> int:
        """Extract port from redis_url."""
        tail = self.redis_url.replace("redis://", "")
        return int(tail.split(":")[1]) if ":" in tail else 6379

    @field_validator("upload_dir", "log_file", mode="before")
    @classmethod
    def _resolve_path(cls, v):
        """Ensure Path fields are always absolute."""
        return Path(v).resolve()


def setup_logging(settings: Settings) -> None:
    """Configure the root logger once at application startup."""
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(settings.log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


# ── Singleton — import this everywhere ──────────────────────────────────────
settings = Settings()
