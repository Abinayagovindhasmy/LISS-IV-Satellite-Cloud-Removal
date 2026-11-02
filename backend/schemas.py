from typing import Optional, List, Any
from datetime import datetime
from pydantic import BaseModel, ConfigDict

class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class ModelMetadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    name: str
    version: str
    training_date: Optional[str] = None
    band_order: Optional[List[str]] = None
    input_channels: Optional[List[str]] = None
    patch_size: Optional[int] = 512
    overlap: Optional[int] = 64

class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    user_id: Optional[int] = None
    filename: str
    original_image: Optional[str] = None
    generated_image: Optional[str] = None
    cloud_mask_image: Optional[str] = None
    comparison_image: Optional[str] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    rmse: Optional[float] = None
    sam: Optional[float] = None
    ergas: Optional[float] = None
    status: str
    progress: int = 0
    error_message: Optional[str] = None
    bounds: Optional[str] = None
    crs: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    cloud_cover_percentage: Optional[float] = None
    processing_time: Optional[float] = None
    model_name: str
    patch_size: int = 512
    overlap: int = 64
    
    # RAG (Retrieval-Augmented Generation) Fields
    use_rag: Optional[bool] = True
    rag_context: Optional[str] = None
    rag_similarity_score: Optional[float] = None

    created_at: datetime

class InferResponse(BaseModel):
    job_id: int
    status: str
    message: str

# RAG Schemas
class RAGQueryRequest(BaseModel):
    prompt: str
    job_id: Optional[int] = None

class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Any]
    retrieved_docs: List[Any]
    spatial_reference: Any
    job_context: Optional[Any] = None
