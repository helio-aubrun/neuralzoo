"""
Schémas Pydantic — Validation des requêtes et réponses de l'API
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    """Une prédiction individuelle dans le top-N."""

    class_name: str = Field(..., description="Nom de la classe animale", example="Chat")
    class_index: int = Field(..., description="Index de la classe (0–5)", example=1, ge=0, le=5)
    confidence: float = Field(
        ..., description="Score de confiance entre 0 et 1", example=0.8731, ge=0.0, le=1.0
    )

    model_config = {"json_schema_extra": {"example": {"class_name": "Chat", "class_index": 1, "confidence": 0.8731}}}


class PredictionResponse(BaseModel):
    """Réponse complète de l'endpoint POST /predict."""

    predicted_class: str = Field(
        ..., description="Classe animale prédite", example="Chat"
    )
    class_index: int = Field(
        ..., description="Index de la classe prédite (0–5)", example=1
    )
    confidence: float = Field(
        ..., description="Score de confiance de la prédiction (0–1)", example=0.8731
    )
    top_predictions: List[TopPrediction] = Field(
        ..., description="Top 3 des prédictions avec leurs scores de confiance"
    )
    inference_time_ms: float = Field(
        ..., description="Temps d'inférence en millisecondes", example=12.45
    )
    image_size: str = Field(
        ..., description="Dimensions originales de l'image envoyée", example="640×480"
    )
    filename: str = Field(
        ..., description="Nom du fichier image envoyé", example="mon_chat.jpg"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_class": "Chat",
                "class_index": 1,
                "confidence": 0.8731,
                "top_predictions": [
                    {"class_name": "Chat",   "class_index": 1, "confidence": 0.8731},
                    {"class_name": "Chien",  "class_index": 3, "confidence": 0.0921},
                    {"class_name": "Oiseau", "class_index": 0, "confidence": 0.0187},
                ],
                "inference_time_ms": 12.45,
                "image_size": "640×480",
                "filename": "mon_chat.jpg",
            }
        }
    }


class HealthResponse(BaseModel):
    """Réponse de l'endpoint GET /health."""

    status: str = Field(..., description="État de l'API : 'ok' ou 'degraded'", example="ok")
    model_loaded: bool = Field(..., description="Indique si le modèle est chargé", example=True)
    model_type: str = Field(..., description="Type de modèle utilisé", example="CNN")
    classes: List[str] = Field(..., description="Liste des classes supportées")
    version: str = Field(..., description="Version de l'API", example="1.0.0")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "model_loaded": True,
                "model_type": "CNN",
                "classes": ["Oiseau", "Chat", "Cerf", "Chien", "Grenouille", "Cheval"],
                "version": "1.0.0",
            }
        }
    }
