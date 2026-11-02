import datetime
from typing import Optional, List
from sqlalchemy import String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.database import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

    jobs: Mapped[List["Job"]] = relationship("Job", back_populates="owner", cascade="all, delete-orphan")

class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_image: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    generated_image: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cloud_mask_image: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    comparison_image: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    psnr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ssim: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rmse: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sam: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ergas: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    status: Mapped[str] = mapped_column(String(50), default="Pending")  # Pending, Running, Completed, Failed
    progress: Mapped[int] = mapped_column(default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    bounds: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # min_lon,min_lat,max_lon,max_lat
    crs: Mapped[Optional[str]] = mapped_column(String(100), default="EPSG:4326")
    width: Mapped[Optional[int]] = mapped_column(nullable=True)
    height: Mapped[Optional[int]] = mapped_column(nullable=True)
    cloud_cover_percentage: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    processing_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_name: Mapped[str] = mapped_column(String(100), default="synthetic_liss4_unet")
    patch_size: Mapped[int] = mapped_column(default=512)
    overlap: Mapped[int] = mapped_column(default=64)

    # RAG (Retrieval-Augmented Generation) Fields
    use_rag: Mapped[bool] = mapped_column(Boolean, default=True)
    rag_context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rag_similarity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))

    owner: Mapped[Optional[User]] = relationship("User", back_populates="jobs")
