import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile

from pipeline.ingest_pipeline import ingest_pipeline

upload_router = APIRouter(prefix="/api/v1", tags=["Upload"])
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {".pdf"}


def _validate_and_save(file: UploadFile) -> Path:
    """Validate a single upload and write it to disk with a safe, unique filename.

    Raises:
        HTTPException 400: if the file type or size is invalid.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Only PDF files are accepted.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Safe filename: uuid prefix prevents collisions and path-traversal attacks
    safe_name = f"{uuid.uuid4()}_{Path(file.filename).name}"
    file_path = UPLOAD_DIR / safe_name

    content = file.file.read()

    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File '{file.filename}' exceeds the {MAX_FILE_SIZE_MB} MB size limit.",
        )

    file_path.write_bytes(content)
    logger.info(f"Saved upload: {safe_name} ({len(content) / 1024:.1f} KB)")
    return file_path


@upload_router.post("/upload")
async def upload(req: Request, background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    saved_files: list[str] = []
    for file in files:
        path = _validate_and_save(file)
        saved_files.append(str(path))

    session_id = str(uuid.uuid4())
    client = req.app.state.client
    embedding_model = req.app.state.embedding_model

    # Run ingestion in the background after the response is sent.
    # FastAPI BackgroundTasks — no external queue or worker process needed.
    background_tasks.add_task(ingest_pipeline, client, embedding_model, saved_files, session_id)

    req.app.state.redis.setex(f"session:{session_id}", 1800, "active")
    logger.info(f"Session {session_id} registered — {len(saved_files)} file(s) queued for ingestion")

    return {
        "session_id": session_id,
        "message": "Upload successful. Ingestion is processing in the background.",
        "documents": len(saved_files),
    }
