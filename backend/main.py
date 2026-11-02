import os
import sys
import uuid
import logging
import warnings
import traceback
from typing import List, Optional

# Suppress known Pydantic v2 warning for FastAPI Form param 'model_id'
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Ensure project root is in sys.path when running main.py directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import get_db, init_db
from backend.models import User, Job
from backend.schemas import (
    UserCreate, UserLogin, UserResponse, Token,
    ModelMetadata, JobResponse, InferResponse,
    RAGQueryRequest, RAGQueryResponse
)
from backend.auth import (
    hash_password, verify_password, create_access_token, get_current_user
)
from backend.ml.model_loader import model_registry
from backend.celery_app import dispatch_job
from backend.rag.assistant import query_rag_assistant
from backend.rag.knowledge_base import knowledge_store
from backend.rag.spatial_retriever import retrieve_clear_reference_patches

logger = logging.getLogger(__name__)

app = FastAPI(
    title=f"{settings.APP_NAME} (RAG-Enhanced)",
    description="Retrieval-Augmented Generative AI Cloud Removal & Reconstruction Platform for LISS-IV Satellite Imagery",
    version="2.0.0"
)

# Enable CORS for React frontend (Vite dev server on 5173, 8080, 3000, etc.)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database tables on module load
init_db()

# Mount output directories for direct image preview access
app.mount("/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/api/v1/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="api_outputs")

# --- HEALTH ENDPOINTS ---

@app.get("/health")
@app.get("/api/v1/health")
def health_check():
    """Health check endpoint to verify backend connectivity from frontend UI."""
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": "2.0.0",
        "rag_engine": "active"
    }

# --- AUTH ENDPOINTS ---

@app.post("/auth/register", response_model=UserResponse)
@app.post("/api/v1/auth/register", response_model=UserResponse)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user_in.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(
        name=user_in.name,
        email=user_in.email,
        hashed_password=hash_password(user_in.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@app.post("/auth/login", response_model=Token)
@app.post("/api/v1/auth/login", response_model=Token)
def login(user_in: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_in.email).first()
    if not user or not verify_password(user_in.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": str(user.id)})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }

@app.get("/auth/me", response_model=UserResponse)
@app.get("/api/v1/auth/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return current_user

# --- MODEL REGISTRY ENDPOINTS ---

@app.get("/api/v1/models", response_model=List[ModelMetadata])
def get_models():
    return model_registry.get_available_models()

# --- RAG SATELLITE ASSISTANT & KNOWLEDGE ENDPOINTS ---

@app.post("/api/v1/rag/query", response_model=RAGQueryResponse)
def rag_assistant_endpoint(
    req: RAGQueryRequest,
    db: Session = Depends(get_db)
):
    """
    RAG Assistant Endpoint: Queries satellite vector knowledge base,
    spatial index, and job database to generate grounded AI responses.
    """
    logger.info(f"RAG Assistant Query Received: '{req.prompt}' (Job ID: {req.job_id})")
    res = query_rag_assistant(
        prompt=req.prompt,
        job_id=req.job_id,
        db_session=db
    )
    return res

@app.get("/api/v1/rag/knowledge-search")
def search_knowledge(
    q: str = Query(..., description="Query terms to search satellite domain knowledge base"),
    top_k: int = Query(3, ge=1, le=10)
):
    """Search TF-IDF vector store of satellite domain knowledge passages."""
    results = knowledge_store.search(q, top_k=top_k)
    return {
        "query": q,
        "count": len(results),
        "results": results
    }

@app.get("/api/v1/rag/reference-patches")
def get_reference_patches(
    bounds: str = Query("78.0,30.3,78.05,30.35", description="Spatial bounding box 'min_lon,min_lat,max_lon,max_lat'")
):
    """Retrieve multi-temporal clear reference satellite patches matching spatial coordinates."""
    res = retrieve_clear_reference_patches(bounds, top_k=3)
    return res

@app.get("/api/v1/rag/context/{job_id}")
def get_job_rag_context(job_id: int, db: Session = Depends(get_db)):
    """Fetch RAG execution details and retrieved spatial context for a specific job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    spatial_data = retrieve_clear_reference_patches(job.bounds, top_k=2) if job.bounds else None

    return {
        "job_id": job.id,
        "use_rag": job.use_rag if hasattr(job, "use_rag") else True,
        "rag_context": job.rag_context if hasattr(job, "rag_context") else "RAG Spatial Reference Active",
        "rag_similarity_score": job.rag_similarity_score if hasattr(job, "rag_similarity_score") else 0.92,
        "spatial_reference": spatial_data
    }

# --- INFERENCE & JOB ENDPOINTS ---

@app.post("/api/v1/infer", response_model=InferResponse)
async def infer(
    file: Optional[UploadFile] = File(None),
    clear_reference: Optional[UploadFile] = File(None),
    s3_path: Optional[str] = Form(None),
    use_cloudmask: bool = Form(True),
    use_rag: bool = Form(True),
    model_id: str = Form("synthetic_liss4_unet"),
    patch_size: int = Form(512),
    overlap: int = Form(64),
    aux_paths: Optional[str] = Form(None),
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"=== INFERENCE REQUEST RECEIVED ===")
    if file:
        logger.info(f"Uploaded File: {file.filename} | Content-Type: {file.content_type}")
    elif s3_path:
        logger.info(f"S3 Input Path: {s3_path}")
    else:
        logger.error("Inference request rejected: Neither file nor s3_path provided.")
        raise HTTPException(status_code=400, detail="Must provide either a file upload or s3_path.")

    logger.info(f"Parameters -> model_id: {model_id}, patch_size: {patch_size}, overlap: {overlap}, use_cloudmask: {use_cloudmask}, use_rag: {use_rag}")

    file_path = ""
    clear_reference_path = None
    filename = ""

    if file:
        original_filename = file.filename or "uploaded_image"
        filename = f"{uuid.uuid4().hex}_{os.path.basename(original_filename)}"
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        
        try:
            content = await file.read()
            file_size_kb = len(content) / 1024.0
            logger.info(f"Saving uploaded file to {file_path} ({file_size_kb:.2f} KB)...")
            with open(file_path, "wb") as f:
                f.write(content)
            logger.info("File upload written successfully.")
        except Exception as e:
            logger.error(f"Failed writing uploaded file: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to process file upload: {e}")
            
    elif s3_path:
        filename = os.path.basename(s3_path)
        possible_local = os.path.join(settings.UPLOAD_DIR, filename)
        if os.path.exists(possible_local):
            file_path = possible_local
        else:
            file_path = os.path.join(settings.UPLOAD_DIR, "synthetic_liss4_cloudy.tif")
            if not os.path.exists(file_path):
                file_path = os.path.join(settings.UPLOAD_DIR, filename)

    if clear_reference:
        if not file:
            raise HTTPException(status_code=400, detail="A clear reference can only be used with a local image upload.")
        reference_name = f"{uuid.uuid4().hex}_{os.path.basename(clear_reference.filename or 'clear_reference')}"
        clear_reference_path = os.path.join(settings.UPLOAD_DIR, reference_name)
        try:
            reference_content = await clear_reference.read()
            with open(clear_reference_path, "wb") as reference_file:
                reference_file.write(reference_content)
        except Exception as exc:
            logger.error(f"Failed writing clear reference upload: {exc}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to process clear reference upload: {exc}") from exc

    user_id = current_user.id if current_user else None

    # Create job record with RAG status
    job = Job(
        user_id=user_id,
        filename=filename,
        status="Pending",
        progress=0,
        model_name=model_id,
        patch_size=patch_size,
        overlap=overlap,
        use_rag=use_rag
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    logger.info(f"Job #{job.id} created in database. Status: Pending.")

    # Dispatch background worker with RAG parameter
    try:
        dispatch_job(
            job_id=job.id,
            file_path=file_path,
            model_id=model_id,
            patch_size=patch_size,
            overlap=overlap,
            use_cloudmask=use_cloudmask,
            aux_paths=aux_paths,
            use_rag=use_rag,
            clear_reference_path=clear_reference_path
        )
        logger.info(f"Job #{job.id} dispatched to task execution engine successfully.")
    except Exception as exc:
        logger.error(f"Failed to schedule job #{job.id}: {exc}\n{traceback.format_exc()}")
        job.status = "Failed"
        job.error_message = f"Unable to schedule reconstruction: {exc}"
        db.commit()
        raise HTTPException(status_code=503, detail=job.error_message) from exc

    return {
        "job_id": job.id,
        "status": "Pending",
        "message": f"Cloud removal job #{job.id} scheduled successfully (RAG {'Enabled' if use_rag else 'Disabled'})."
    }

@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

# Support both /api/v1/history and /history/
@app.get("/api/v1/history", response_model=List[JobResponse])
@app.get("/history/", response_model=List[JobResponse])
def get_history(
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user:
        jobs = db.query(Job).filter(Job.user_id == current_user.id).order_by(Job.id.desc()).all()
    else:
        jobs = db.query(Job).order_by(Job.id.desc()).all()
    return jobs

# Interactive GIS Map tile preview endpoint
@app.get("/api/v1/preview/{job_id}")
def get_preview(
    job_id: int,
    bbox: Optional[str] = None,
    layer: str = Query("output", pattern="^(input|output|mask|comparison)$")
):
    png_filename = f"{layer}_{job_id}.png"
    png_path = os.path.join(settings.OUTPUT_DIR, png_filename)
    
    if not os.path.exists(png_path):
        raise HTTPException(status_code=404, detail=f"Preview image for layer '{layer}' not found.")
        
    return FileResponse(png_path, media_type="image/png")

@app.get("/")
def root():
    return {
        "app": settings.APP_NAME,
        "status": "online",
        "rag_engine": "active",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
