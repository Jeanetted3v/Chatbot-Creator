
"""To run:
python -m src.backend.api.main_data_ingest
Interact via SwaggerUi: http://localhost:8001/ingest/docs
Check health: curl http://localhost:8001/health
"""
import os
from datetime import datetime
import logging
import uvicorn
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from hydra import initialize, compose
from fastapi import APIRouter, UploadFile, File
from src.backend.utils.logging import setup_logging
from src.backend.dataloaders.local_doc_loader import load_local_doc
from src.backend.dataprocessor.chunker import batch_chunk_doc
from src.backend.dataprocessor.embedder import embed_doc

setup_logging()
logger = logging.getLogger(__name__)

router = APIRouter()
ORIGINS = ["*"]


# -----------------------
# Config loading
# -----------------------
def get_config():
    """Dependency to provide Hydra configuration."""
    with initialize(version_base=None, config_path="./../../../config"):
        cfg = compose(config_name="data_ingest.yaml", return_hydra_config=True)
    return cfg


cfg = get_config()

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(
    title="Data Ingest Pipeline for Chatbot",
    description="API for uploading and embedding documents into ChromaDB.",
    version="1.0",
    docs_url="/ingest/docs",
    openapi_url="/ingest/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,   # change to ORIGINS in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# Upload Endpoint
# -----------------------
@router.post("/upload_doc")
async def upload_doc(files: List[UploadFile] = File(...)):
    """Endpoint for uploading and embedding a document into ChromaDB."""
    try:
        # 1. Save the uploaded file
        upload_dir = os.path.join(cfg.data_dir, "data_to_ingest")
        os.makedirs(upload_dir, exist_ok=True)

        all_chunked_docs = []
        uploaded_paths = []
        
        for file in files:
            saved_path = os.path.join(upload_dir, file.filename)
            with open(saved_path, "wb") as f:
                f.write(await file.read())
            logger.info(f"Uploaded file saved to {saved_path}")
            uploaded_paths.append({"path": saved_path})

            # 2. Load and chunk the document
            cfg.local_doc.paths = uploaded_paths
            loaded_docs = load_local_doc(cfg)
            chunked_docs = batch_chunk_doc(cfg, loaded_docs)
            all_chunked_docs.extend(chunked_docs)

        # 3. Embed and store using ChromaDB
        await embed_doc(cfg, chunked_docs)

        return {
            "message": f"{len(files)} files embedded and stored successfully.",
            "filenames": [file.filename for file in files],
        }
    
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        return {"error": str(e)}


# -----------------------
# Health check
# -----------------------
@app.get("/")
@app.head("/")
async def root():
    """Root endpoint for the FastAPI server."""
    return {
        "message": "Welcome to the Document Ingestion API",
        "version": 1.0,
        "docs": "ingest/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for the FastAPI server."""
    return {"status": "healthy"}


# -----------------------
# Register router
# -----------------------
app.include_router(router, prefix="/ingest", tags=["data-ingest"])


def main() -> None:
    """Main function to run the FastAPI server."""
    uvicorn.run(
        "src.backend.api.main_data_ingest:app",
        host="0.0.0.0",
        port=8080,
        reload=cfg.api.reload if hasattr(cfg.api, "reload") else False,
    )


if __name__ == "__main__":
    main()
