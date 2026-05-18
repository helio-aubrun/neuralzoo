"""
Chargeur de modèle — ModelLoader
==================================
Gère le chargement du modèle Keras entraîné.
En mode démonstration (si aucun modèle n'est trouvé), utilise
un modèle CNN léger recréé à la volée pour permettre les tests.
"""

import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Chemins possibles pour le modèle sauvegardé
MODEL_PATHS = [
    Path("model/best_cnn.keras"),
    Path("model/best_mlp.keras"),
    Path("best_cnn.keras"),
    Path("best_mlp.keras"),
]

NUM_CLASSES = 6


class ModelLoader:
    """Charge et encapsule le modèle de prédiction."""

    def __init__(self):
        self._model = None
        self._model_type = "non chargé"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_type(self) -> str:
        return self._model_type

    def load(self):
        """Tente de charger le modèle sauvegardé, sinon crée un modèle de démonstration."""
        # ── Tentative de chargement d'un modèle existant ────────────────────
        try:
            import tensorflow as tf
            for path in MODEL_PATHS:
                if path.exists():
                    logger.info(f"Chargement du modèle : {path}")
                    self._model = tf.keras.models.load_model(str(path))
                    self._model_type = "CNN" if "cnn" in path.name.lower() else "MLP"
                    logger.info(f"✅ Modèle '{self._model_type}' chargé depuis {path}")
                    return

            # ── Aucun modèle trouvé → mode démo ──────────────────────────────
            logger.warning("⚠️  Aucun modèle sauvegardé trouvé → création d'un modèle de démonstration.")
            self._model = self._build_demo_cnn()
            self._model_type = "CNN (démo — non entraîné)"

        except ImportError:
            logger.error("❌ TensorFlow non installé. Installez-le avec : pip install tensorflow")
            raise
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement : {e}")
            raise

    def _build_demo_cnn(self):
        """
        Construit un CNN léger identique à celui du notebook.
        Utilisé uniquement si aucun modèle entraîné n'est disponible.
        """
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential([
            keras.Input(shape=(32, 32, 3)),
            layers.Rescaling(1.0 / 255),

            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ], name="CNN_demo")

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info("Modèle CNN de démonstration créé (poids aléatoires).")
        return model

    def predict(self, img_batch: np.ndarray) -> np.ndarray:
        """
        Lance l'inférence sur un batch d'images.

        Args:
            img_batch: array numpy de shape (N, 32, 32, 3) en float32

        Returns:
            array de probabilités de shape (N, NUM_CLASSES)
        """
        if not self.is_loaded:
            raise RuntimeError("Le modèle n'est pas chargé. Appelez load() d'abord.")
        return self._model.predict(img_batch, verbose=0)
