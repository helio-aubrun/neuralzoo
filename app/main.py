"""
CIFAR-10 Animal Classifier — API FastAPI
========================================
Endpoint principal : POST /predict
  - Reçoit une image (multipart/form-data)
  - Retourne la classe prédite + score de confiance + top-3

Swagger UI disponible sur : http://localhost:8000/docs
ReDoc disponible sur      : http://localhost:8000/redoc
"""

import io
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import PredictionResponse, HealthResponse, TopPrediction
from app.model_loader import ModelLoader

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Constantes ───────────────────────────────────────────────────────────────
ANIMAL_CLASSES = ["Oiseau", "Chat", "Cerf", "Chien", "Grenouille", "Cheval"]
IMG_SIZE       = (32, 32)
MAX_FILE_SIZE  = 10 * 1024 * 1024   # 10 Mo
ALLOWED_TYPES  = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

# ── Chargement du modèle au démarrage (lifespan) ─────────────────────────────
model_loader = ModelLoader()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Démarrage de l'API — chargement du modèle…")
    model_loader.load()
    logger.info("✅ Modèle chargé avec succès.")
    yield
    logger.info("🛑 Arrêt de l'API.")

# ── Création de l'application ─────────────────────────────────────────────────
app = FastAPI(
    title="🐾 CIFAR-10 Animal Classifier API",
    description="""
## API de classification d'images d'animaux

Cette API utilise un modèle **CNN** entraîné sur le dataset **CIFAR-10** (sous-ensemble animaux)
pour classer des images dans l'une des 6 catégories suivantes :

| Classe | Emoji |
|--------|-------|
| Oiseau | 🐦 |
| Chat   | 🐱 |
| Cerf   | 🦌 |
| Chien  | 🐶 |
| Grenouille | 🐸 |
| Cheval | 🐴 |

### Utilisation
1. Envoyez une image via `POST /predict`
2. Recevez la classe prédite, le score de confiance et le top-3

### Notes
- Format accepté : JPEG, PNG, WebP, BMP
- Taille maximale : 10 Mo
- L'image est redimensionnée à 32×32 pixels avant l'inférence
    """,
    version="1.0.0",
    contact={
        "name": "Équipe Deep Learning",
        "email": "contact@example.com",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
)

# ── CORS (pour le client Streamlit) ──────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # En prod : restreindre à l'URL du client
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/",
    summary="Accueil",
    description="Redirige vers la documentation Swagger.",
    tags=["Général"],
)
async def root():
    return {
        "message": "🐾 CIFAR-10 Animal Classifier API — voir /docs pour la documentation.",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="État de santé de l'API",
    description="Vérifie que l'API est opérationnelle et que le modèle est chargé.",
    tags=["Monitoring"],
)
async def health():
    return HealthResponse(
        status="ok" if model_loader.is_loaded else "degraded",
        model_loaded=model_loader.is_loaded,
        model_type=model_loader.model_type,
        classes=ANIMAL_CLASSES,
        version="1.0.0",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Prédire la classe d'une image animale",
    description="""
Envoie une image et reçoit :
- **predicted_class** : la classe prédite (ex. `Chat`)
- **confidence** : score de confiance entre 0 et 1
- **top_predictions** : les 3 meilleures prédictions avec leurs scores
- **inference_time_ms** : temps d'inférence en millisecondes

L'image est automatiquement redimensionnée à **32×32 pixels** (taille CIFAR-10).
    """,
    tags=["Prédiction"],
    responses={
        200: {"description": "Prédiction réussie"},
        400: {"description": "Image invalide ou format non supporté"},
        503: {"description": "Modèle non disponible"},
    },
)
async def predict(
    file: UploadFile = File(
        ...,
        description="Image à classifier (JPEG, PNG, WebP, BMP — max 10 Mo)",
    )
):
    # ── Vérification du modèle ────────────────────────────────────────────────
    if not model_loader.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modèle n'est pas encore chargé. Veuillez réessayer dans quelques secondes.",
        )

    # ── Vérification du type MIME ─────────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Type de fichier non supporté : '{file.content_type}'. "
                   f"Types acceptés : {', '.join(ALLOWED_TYPES)}",
        )

    # ── Lecture et vérification de la taille ──────────────────────────────────
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Fichier trop volumineux ({len(contents)/1024/1024:.1f} Mo). Maximum : 10 Mo.",
        )

    # ── Décodage et prétraitement de l'image ──────────────────────────────────
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_resized = image.resize(IMG_SIZE, Image.LANCZOS)
        img_array = np.array(image_resized, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)   # (1, 32, 32, 3)
    except Exception as e:
        logger.error(f"Erreur décodage image : {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Impossible de lire l'image : {str(e)}",
        )

    # ── Inférence ─────────────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        probabilities = model_loader.predict(img_batch)[0]   # shape: (6,)
        inference_ms  = round((time.perf_counter() - t0) * 1000, 2)
    except Exception as e:
        logger.error(f"Erreur inférence : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'inférence : {str(e)}",
        )

    # ── Construction de la réponse ────────────────────────────────────────────
    top3_idx = np.argsort(probabilities)[::-1][:3]
    top_predictions = [
        TopPrediction(
            class_name=ANIMAL_CLASSES[i],
            class_index=int(i),
            confidence=round(float(probabilities[i]), 4),
        )
        for i in top3_idx
    ]

    predicted_idx = int(np.argmax(probabilities))

    logger.info(
        f"Prédiction : {ANIMAL_CLASSES[predicted_idx]} "
        f"({probabilities[predicted_idx]*100:.1f}%) | "
        f"Fichier : {file.filename} | "
        f"Durée : {inference_ms} ms"
    )

    return PredictionResponse(
        predicted_class=ANIMAL_CLASSES[predicted_idx],
        class_index=predicted_idx,
        confidence=round(float(probabilities[predicted_idx]), 4),
        top_predictions=top_predictions,
        inference_time_ms=inference_ms,
        image_size=f"{image.width}×{image.height}",
        filename=file.filename or "inconnu",
    )
