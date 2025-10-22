from functools import lru_cache
from pathlib import Path
from typing import Iterable

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "iris_model.pkl"


@lru_cache(maxsize=1)
def _load_model():
    """Load and cache the trained model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def predict_data(X: Iterable[Iterable[float]]):
    """Return class predictions for the provided samples."""
    model = _load_model()
    return model.predict(list(X))


def predict_probabilities(X: Iterable[Iterable[float]]):
    """Return class probabilities when available."""
    model = _load_model()
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Loaded model does not support probability predictions")
    return model.predict_proba(list(X))
