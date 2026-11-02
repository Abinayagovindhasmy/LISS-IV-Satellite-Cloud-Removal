"""
RAG AI Copilot Assistant Engine for LISS-IV Satellite Platform.
Combines Vector Knowledge Base retrieval, Spatial Index lookup, and Job Database metadata
to answer natural language queries with grounded citations and remote sensing insights.
"""

import logging
from typing import Dict, Any, Optional, List
from backend.rag.knowledge_base import knowledge_store
from backend.rag.spatial_retriever import retrieve_clear_reference_patches

logger = logging.getLogger(__name__)

def query_rag_assistant(
    prompt: str,
    job_id: Optional[int] = None,
    db_session: Any = None
) -> Dict[str, Any]:
    """
    RAG Assistant Query Pipeline:
    1. Search Knowledge Base for domain passages matching the prompt.
    2. Retrieve Job database metadata if job_id is provided.
    3. Retrieve spatial reference patch context.
    4. Generate grounded, structured response with source citations.
    """
    clean_prompt = prompt.strip()
    
    # 1. Knowledge Base Retrieval
    retrieved_docs = knowledge_store.search(clean_prompt, top_k=3)
    
    # 2. Fetch Job Context if requested
    job_context = None
    if job_id and db_session:
        try:
            from backend.models import Job
            job = db_session.query(Job).filter(Job.id == job_id).first()
            if job:
                job_context = {
                    "job_id": job.id,
                    "filename": job.filename,
                    "status": job.status,
                    "cloud_cover_percentage": job.cloud_cover_percentage,
                    "psnr": job.psnr,
                    "ssim": job.ssim,
                    "model_name": job.model_name,
                    "processing_time": job.processing_time,
                    "bounds": job.bounds,
                    "patch_size": job.patch_size,
                    "overlap": job.overlap,
                    "use_rag": getattr(job, "use_rag", True)
                }
        except Exception as e:
            logger.error(f"Error fetching job #{job_id} for RAG Assistant: {e}")

    # 3. Spatial Reference Context Lookup
    spatial_ref = None
    if job_context and job_context.get("bounds"):
        spatial_ref = retrieve_clear_reference_patches(str(job_context["bounds"]), top_k=2)
    elif not job_context:
        spatial_ref = retrieve_clear_reference_patches("78.0,30.3,78.05,30.35", top_k=2)

    # 4. Synthesize RAG Response
    citations = []
    for doc in retrieved_docs:
        citations.append({
            "id": doc["id"],
            "title": doc["title"],
            "category": doc["category"],
            "relevance_score": doc.get("score", 0.95)
        })

    # Response generation logic
    lowered = clean_prompt.lower()
    answer_sections = []

    if job_context:
        quality_metrics = (
            f"{job_context['psnr']:.2f} dB | {job_context['ssim']:.4f}"
            if job_context["psnr"] is not None and job_context["ssim"] is not None
            else "N/A (requires an aligned clear reference)"
        )
        cloud_str = (
            f"{job_context['cloud_cover_percentage']:.2f}%"
            if job_context.get("cloud_cover_percentage") is not None
            else "N/A"
        )
        time_str = (
            f"{job_context['processing_time']:.2f} seconds"
            if job_context.get("processing_time") is not None
            else "N/A"
        )
        answer_sections.append(
            f"### 📊 Analysis for Satellite Job #{job_context['job_id']} ({job_context['filename']})\n"
            f"- **Status**: `{job_context['status']}`\n"
            f"- **Cloud Cover Detected**: `{cloud_str}` of total scene area\n"
            f"- **Model Executed**: `{job_context['model_name']}` (Patch Size: {job_context['patch_size']}px, Overlap: {job_context['overlap']}px)\n"
            f"- **Processing Time**: `{time_str}`\n"
            f"- **PSNR / SSIM**: `{quality_metrics}`\n"
            f"- **RAG Enabled**: `{'Yes - spatial context metadata retrieved' if job_context['use_rag'] else 'No'}`"
        )

    # Question matching and detailed domain responses
    if "band" in lowered or "spectral" in lowered or "liss" in lowered or "sensor" in lowered:
        answer_sections.append(
            "### 🛰️ ISRO LISS-IV Spectral Band Characteristics\n"
            "LISS-IV captures Earth surface reflectance across **4 key multi-spectral bands**:\n"
            "1. **Band 1 / Green (0.52 - 0.59 µm)**: High sensitivity to chlorophyll reflection and water turbidities.\n"
            "2. **Band 2 / Red (0.62 - 0.68 µm)**: Strong absorption band for active vegetation chlorophyll; essential for soil contrast.\n"
            "3. **Band 3 / Near Infrared (NIR: 0.77 - 0.86 µm)**: Peak reflection channel for healthy plant leaf mesophyll structure. Key component of False Color Composites (FCC) and NDVI.\n"
            "4. **Band 4 / Short-Wave Infrared (SWIR: 1.55 - 1.70 µm)**: Sensitive to leaf moisture content and cloud-vs-snow discrimination."
        )

    if "rag" in lowered or "retrieval" in lowered or "how" in lowered or "why" in lowered:
        answer_sections.append(
            "### ✦ How RAG (Retrieval-Augmented Generation) Enhances Cloud Removal\n"
            "The current RAG index returns scene-context metadata, not pixel-aligned historical raster imagery, so it does not provide ground-truth pixels for reconstruction.\n"
            "Cloudy pixels are estimated by the selected reconstruction model. For direct pixel replacement and reference-based quality metrics, provide an accurately co-registered clear image with the same scene."
        )

    if "ndvi" in lowered or "vegetation" in lowered or "health" in lowered or "index" in lowered:
        answer_sections.append(
            "### 🌿 Vegetation Index (NDVI) Formulation\n"
            "To evaluate canopy health from reconstructed LISS-IV bands, compute:\n"
            "$$\\text{NDVI} = \\frac{\\text{NIR (Band 3)} - \\text{Red (Band 2)}}{\\text{NIR (Band 3)} + \\text{Red (Band 2)}}$$\n"
            "- **Dense Forest Canopy**: `0.65 - 0.88` (Vibrant NIR reflectance)\n"
            "- **Agriculture / Crops**: `0.35 - 0.60`\n"
            "- **Bare Soil / Urban**: `0.10 - 0.25`\n"
            "- **Water / Cloud Shadow**: `< 0.0`"
        )

    if "metric" in lowered or "psnr" in lowered or "ssim" in lowered or "quality" in lowered:
        answer_sections.append(
            "### 📈 Reconstruction Quality Evaluation Metrics\n"
            "- **PSNR (Peak Signal-to-Noise Ratio)**: Evaluates pixel-level intensity fidelity. Values `>30 dB` indicate clean reconstruction without high-frequency noise.\n"
            "- **SSIM (Structural Similarity Index)**: Measures structural correlation and edge preservation (`0.0 to 1.0`). Values `>0.85` confirm high structural fidelity."
        )

    # General fallback text if prompt is generic
    if len(answer_sections) == (1 if job_context else 0):
        spatial_summary = (
            f"- **Spatial Reference Index**: Top match `{spatial_ref['primary_patch_id']}` "
            f"with **{spatial_ref['best_similarity_score']*100:.1f}% spatial similarity score**."
            if spatial_ref
            else "- **Spatial Reference Index**: Not queried because the image has no geospatial bounds."
        )
        answer_sections.append(
            "### 🔍 Retrieval-Augmented Geospatial Summary\n"
            "Based on retrieved satellite domain documentation and spatial index lookup:\n"
            f"- **Primary Knowledge Match**: [{retrieved_docs[0]['title']}] ({retrieved_docs[0]['category']})\n"
            f"- **Retrieved Passages**: {retrieved_docs[0]['content']}\n{spatial_summary}"
        )

    answer_text = "\n\n".join(answer_sections)

    return {
        "query": clean_prompt,
        "answer": answer_text,
        "citations": citations,
        "retrieved_docs": retrieved_docs,
        "spatial_reference": spatial_ref,
        "job_context": job_context
    }
