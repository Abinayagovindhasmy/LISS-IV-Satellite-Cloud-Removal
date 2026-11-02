import os
import logging
import threading
from typing import Optional

try:
    from celery import Celery
    HAS_CELERY = True
except ImportError:
    Celery = None
    HAS_CELERY = False

from backend.config import settings
from backend.database import SessionLocal
from backend.models import Job
from backend.ml.inference import run_cloud_removal_pipeline

logger = logging.getLogger(__name__)

celery_app = None
if HAS_CELERY and Celery is not None:
    celery_app = Celery(
        "cloudvision_tasks",
        broker=settings.REDIS_URL,
        backend=settings.REDIS_URL
    )
    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
    )

def execute_job_task(
    job_id: int,
    file_path: str,
    model_id: str,
    patch_size: int,
    overlap: int,
    use_cloudmask: bool,
    aux_paths: Optional[str] = None,
    use_rag: bool = True,
    clear_reference_path: Optional[str] = None
):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"Job #{job_id} not found in database.")
            return

        job.status = "Running"
        job.progress = 5
        job.use_rag = use_rag
        db.commit()

        def update_progress(pct: int):
            try:
                if job is not None:
                    job.progress = pct
                    db.commit()
            except Exception as e:
                logger.error(f"Failed to update progress for job #{job_id}: {e}")

        res = run_cloud_removal_pipeline(
            input_file_path=file_path,
            job_id=job_id,
            model_id=model_id,
            patch_size=patch_size,
            overlap=overlap,
            use_cloudmask=use_cloudmask,
            aux_paths=aux_paths,
            clear_reference_path=clear_reference_path,
            use_rag=use_rag,
            progress_callback=update_progress
        )

        job.status = "Completed"
        job.progress = 100
        job.original_image = res["original_image"]
        job.generated_image = res["generated_image"]
        job.cloud_mask_image = res["cloud_mask_image"]
        job.comparison_image = res.get("comparison_image")
        job.psnr = res.get("psnr")
        job.ssim = res.get("ssim")
        job.rmse = res.get("rmse")
        job.sam = res.get("sam")
        job.ergas = res.get("ergas")
        job.bounds = res["bounds"]
        job.crs = res["crs"]
        job.width = res["width"]
        job.height = res["height"]
        job.cloud_cover_percentage = res["cloud_cover_percentage"]
        job.processing_time = res["processing_time"]

        # RAG Persistence
        job.use_rag = res.get("use_rag", True)
        job.rag_context = res.get("rag_context")
        job.rag_similarity_score = res.get("rag_similarity_score")

        db.commit()

    except Exception as e:
        logger.exception(f"Job #{job_id} failed with error: {e}")
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "Failed"
            job.error_message = str(e)
            db.commit()
    finally:
        db.close()

if HAS_CELERY and celery_app is not None:
    @celery_app.task(name="backend.celery_app.process_cloud_removal_job")
    def process_cloud_removal_job(
        job_id: int,
        file_path: str,
        model_id: str,
        patch_size: int,
        overlap: int,
        use_cloudmask: bool,
        aux_paths: Optional[str] = None,
        use_rag: bool = True,
        clear_reference_path: Optional[str] = None
    ):
        execute_job_task(job_id, file_path, model_id, patch_size, overlap, use_cloudmask, aux_paths, use_rag, clear_reference_path)

def dispatch_job(
    job_id: int,
    file_path: str,
    model_id: str,
    patch_size: int,
    overlap: int,
    use_cloudmask: bool,
    aux_paths: Optional[str] = None,
    use_rag: bool = True,
    clear_reference_path: Optional[str] = None
):
    """
    Dispatches task via Celery if Celery & Redis are reachable, or falls back to background Thread.
    """
    dispatched = False
    if HAS_CELERY and celery_app is not None:
        try:
            celery_app.control.inspect().ping()
            process_cloud_removal_job.delay(
                job_id, file_path, model_id, patch_size, overlap, use_cloudmask, aux_paths, use_rag, clear_reference_path
            )
            logger.info(f"Dispatched job #{job_id} via Celery worker queue (RAG={use_rag}).")
            dispatched = True
        except Exception as e:
            logger.info(f"Celery queue unavailable ({e}). Falling back to background thread.")
    
    if not dispatched:
        thread = threading.Thread(
            target=execute_job_task,
            args=(job_id, file_path, model_id, patch_size, overlap, use_cloudmask, aux_paths, use_rag, clear_reference_path),
            daemon=True
        )
        thread.start()
        logger.info(f"Started job #{job_id} in background thread (RAG={use_rag}).")
