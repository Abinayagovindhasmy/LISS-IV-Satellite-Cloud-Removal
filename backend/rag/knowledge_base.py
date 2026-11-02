"""
Satellite RAG Knowledge Base & Vector Retrieval Engine.
Provides domain knowledge retrieval on LISS-IV Satellite Sensors, Cloud Removal Algorithms,
Spectral Indices (NDVI, NDWI), and Spatial-Temporal Image Reconstruction.
"""

import math
import re
from typing import List, Dict, Any, Tuple

# Domain Knowledge Corpus for LISS-IV Satellite Imagery and RAG Spatial-Temporal Reconstruction
LISS4_KNOWLEDGE_CORPUS = [
    {
        "id": "liss4_specs",
        "title": "ISRO Resourcesat LISS-IV Sensor Specifications",
        "category": "Sensor Specs",
        "content": (
            "LISS-IV (Linear Imaging Self-Scanning Sensor IV) is a high-resolution multi-spectral camera "
            "aboard ISRO's Resourcesat-1 (IRS-P6), Resourcesat-2, and Resourcesat-2A satellites. "
            "It operates in 3 spectral bands in Mono mode (5.8 meter spatial resolution) and MX mode: "
            "Band 2 (Green: 0.52 - 0.59 µm), Band 3 (Red: 0.62 - 0.68 µm), and Band 4 (Near Infrared / NIR: 0.77 - 0.86 µm), "
            "with an optional Short-Wave Infrared (SWIR: 1.55 - 1.70 µm) band. "
            "Swath width is 23 km in MX mode and 70 km in Mono mode. Revisit capability is 5 days with steerability up to ±26°."
        )
    },
    {
        "id": "rag_cloud_removal",
        "title": "Retrieval-Augmented Generation (RAG) for Satellite Cloud Removal",
        "category": "RAG Architecture",
        "content": (
            "RAG (Retrieval-Augmented Generation) in satellite image reconstruction combines multi-temporal geospatial "
            "retrieval with generative neural networks (U-Net / Spatio-Temporal Diffusion / GANs). "
            "A spatial index can return metadata about potentially relevant clear scenes, but metadata alone is not a pixel-aligned raster "
            "and must not be described as ground truth or blended into cloud-covered pixels. For reference-based reconstruction, provide "
            "an accurately co-registered clear image; otherwise the model's cloudy-pixel output is an estimate."
        )
    },
    {
        "id": "spectral_indices",
        "title": "Spectral Indices: NDVI and Vegetation Analysis in LISS-IV",
        "category": "Remote Sensing",
        "content": (
            "In LISS-IV imagery, vegetation vigor and canopy health are calculated using the Normalized Difference Vegetation Index (NDVI): "
            "NDVI = (NIR - Red) / (NIR + Red) = (Band 4 - Band 3) / (Band 4 + Band 3). "
            "Healthy dense vegetation absorbs Red light (high chlorophyll absorption) and strongly reflects NIR light, yielding NDVI values between 0.6 and 0.9. "
            "Water bodies exhibit negative NDVI (-0.1 to -0.5), while bare soil and urban built-up areas show low positive values (0.1 to 0.3)."
        )
    },
    {
        "id": "cloud_detection_thresholding",
        "title": "Automated Cloud and Cloud-Shadow Detection",
        "category": "Preprocessing",
        "content": (
            "Automated cloud mask generation for LISS-IV uses multi-band thresholding across visible and infrared channels. "
            "Clouds exhibit high reflectance in Green (Band 2) and Red (Band 3) wavelengths while maintaining high top-of-atmosphere brightness. "
            "Cloud masks are calculated by evaluating composite reflectance thresholds (>0.60 across Green/Red) combined with SWIR/NIR ratio tests. "
            "Cloud shadows are identified by low NIR reflectance adjacent to detected cloud vectors aligned with solar azimuth."
        )
    },
    {
        "id": "tiling_and_hanning_windowing",
        "title": "Windowed Tiling & Hanning Overlap Blending",
        "category": "Inference Optimization",
        "content": (
            "To process large high-resolution GeoTIFF satellite scenes (e.g. 4096 x 4096 pixels) without GPU Out-Of-Memory (OOM) errors, "
            "the inference pipeline breaks the input array into overlapping patches (e.g., 512x512 with 64px overlap). "
            "Border discontinuities and seams between adjacent tile predictions are eliminated by applying a 2D Hanning window (raised cosine) weighting grid. "
            "Reconstructed patches are accumulated with weighted normalization to ensure seamless spatial continuity across patch boundaries."
        )
    },
    {
        "id": "evaluation_metrics",
        "title": "Quality Assessment Metrics: PSNR and SSIM",
        "category": "Quality Control",
        "content": (
            "The reconstruction quality of cloud removal is evaluated using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM). "
            "PSNR measures spatial signal fidelity in decibels (dB), where values above 30 dB indicate high reconstruction accuracy. "
            "SSIM measures luminance, contrast, and structural correlation between reconstructed and ground truth clear images, with values ranging from 0 to 1.0 (where 1.0 represents perfect structural match)."
        )
    },
    {
        "id": "isro_resourcesat_orbit",
        "title": "Resourcesat Mission & Orbit Parameters",
        "category": "Satellite Orbit",
        "content": (
            "ISRO's Resourcesat satellites operate in a Sun-Synchronous Polar Orbit at an altitude of approximately 817 km, "
            "with an orbital inclination of 98.69° and an equatorial local cross time of 10:30 AM (descending node). "
            "This consistent solar illumination geometry makes multi-temporal LISS-IV image retrieval highly suitable for RAG reference alignment."
        )
    }
]

def tokenize(text: str) -> List[str]:
    """Tokenize and normalize text into words."""
    words = re.findall(r'\w+', text.lower())
    # Exclude basic stopwords
    stopwords = {"a", "an", "the", "in", "on", "of", "to", "for", "is", "are", "and", "or", "with", "this", "that", "by", "from", "at", "as", "it"}
    return [w for w in words if w not in stopwords and len(w) > 1]

class KnowledgeVectorStore:
    """TF-IDF and Cosine Similarity Vector Store for RAG Knowledge Retrieval."""
    
    def __init__(self, corpus: List[Dict[str, Any]]):
        self.corpus = corpus
        self.vocab: Dict[str, int] = {}
        self.doc_vectors: List[Dict[int, float]] = []
        self.idf: Dict[str, float] = {}
        self._build_index()

    def _build_index(self):
        # 1. Collect all terms
        doc_count = len(self.corpus)
        df: Dict[str, int] = {}
        doc_tokens_list = []

        for doc in self.corpus:
            full_text = f"{doc['title']} {doc['category']} {doc['content']}"
            tokens = tokenize(full_text)
            doc_tokens_list.append(tokens)
            unique_terms = set(tokens)
            for t in unique_terms:
                df[t] = df.get(t, 0) + 1

        # 2. Build vocab & IDF
        vocab_idx = 0
        for term, freq in df.items():
            self.vocab[term] = vocab_idx
            vocab_idx += 1
            self.idf[term] = math.log((doc_count + 1) / (freq + 0.5)) + 1.0

        # 3. Vectorize documents
        for tokens in doc_tokens_list:
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            
            vec: Dict[int, float] = {}
            length = len(tokens) or 1
            for term, count in tf.items():
                if term in self.vocab:
                    t_idx = self.vocab[term]
                    vec[t_idx] = (count / length) * self.idf[term]
            
            # Normalize vector
            norm = math.sqrt(sum(val ** 2 for val in vec.values())) or 1.0
            norm_vec = {k: v / norm for k, v in vec.items()}
            self.doc_vectors.append(norm_vec)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search corpus using cosine similarity."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return self.corpus[:top_k]

        tf: Dict[str, int] = {}
        for t in query_tokens:
            tf[t] = tf.get(t, 0) + 1
        
        q_vec: Dict[int, float] = {}
        length = len(query_tokens)
        for term, count in tf.items():
            if term in self.vocab:
                t_idx = self.vocab[term]
                q_vec[t_idx] = (count / length) * self.idf[term]
        
        norm = math.sqrt(sum(val ** 2 for val in q_vec.values())) or 1.0
        q_norm_vec = {k: v / norm for k, v in q_vec.items()}

        scores = []
        for idx, d_vec in enumerate(self.doc_vectors):
            dot_product = sum(val * d_vec.get(k, 0.0) for k, val in q_norm_vec.items())
            scores.append((dot_product, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, idx in scores[:top_k]:
            doc = dict(self.corpus[idx])
            doc["score"] = round(float(score), 4)
            results.append(doc)

        return results

# Global Knowledge Base instance
knowledge_store = KnowledgeVectorStore(LISS4_KNOWLEDGE_CORPUS)
