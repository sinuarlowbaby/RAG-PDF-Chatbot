from fastapi import APIRouter,UploadFile,File,Request
import uuid
from pipeline.ingest_pipeline import ingest_pipeline

import os
from retrieval.hybrid_document_retrieval import initialize_retrievers
import logging

upload_router = APIRouter(prefix="/api/v1", tags=["Upload"])
UPLOAD_DIR = "uploads"
logger = logging.getLogger(__name__)

def save_files(files)-> list[str]:
    file_paths = []
    os.makedirs(UPLOAD_DIR,exist_ok=True)
    for file in files:
        file_path = os.path.join(UPLOAD_DIR,file.filename)
        with open(file_path,"wb") as f:
            f.write(file.file.read())
        file_paths.append(file_path)
    return file_paths


@upload_router.post("/upload")
async def upload(req: Request, files: list[UploadFile] = File(...)):
    session_id = str(uuid.uuid4())
    client = req.app.state.client
    embedding_model = req.app.state.embedding_model
    saved_files = save_files(files)
    vector_store,doc_chunks = ingest_pipeline(client,embedding_model,saved_files,session_id)

    req.app.state.redis.setex(f"session:{session_id}", 1800, "active")
    logger.info(f"Session {session_id} registered to redis")
    
    if not hasattr(req.app.state, "sessions"):
        logger.info("No session map found, creating one")
        req.app.state.sessions = {}

    # req.app.state.sessions[session_id] = {
    #     "vector_store": vector_store,
    #     "documents": doc_chunks,
    #     "hybrid_retriever": initialize_retrievers(vector_store,doc_chunks,session_id)
    # }

    return {"session_id":session_id,"message":"Upload successful","documents":len(saved_files)}