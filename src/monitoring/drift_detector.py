import numpy as np
import logging

logger = logging.getLogger(__name__)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - np.dot(a, b) / (norm_a * norm_b)

def detect_embedding_drift(
    existing_embeddings: list,
    new_embeddings: list,
    threshold: float = 0.15
) -> dict:
    if existing_embeddings is None or len(existing_embeddings) == 0:
        return {"drift_detected": False, "distance": 0.0, "message": "No baseline yet."}

    if new_embeddings is None or len(new_embeddings) == 0:
        return {"drift_detected": False, "distance": 0.0, "message": "No new embeddings."}
    
    existing_embeddings = existing["embeddings"] if existing["embeddings"] is not None else []
    new_centroid = np.mean(np.array(new_embeddings), axis=0)
    distance = cosine_distance(existing_centroid, new_centroid)
    drift_detected = distance > threshold
    
    result = {
        "drift_detected": drift_detected,
        "distance": round(distance, 4),
        "threshold": threshold,
        "existing_count": len(existing_embeddings),
        "new_count": len(new_embeddings),
        "message": (
            f"⚠️ Drift detected: distance={distance:.4f} > threshold={threshold}"
            if drift_detected else
            f"✅ No drift: distance={distance:.4f} within threshold={threshold}"
        )
    }
    
    if drift_detected:
        logger.warning(result["message"])
    else:
        logger.info(result["message"])
    
    return result
