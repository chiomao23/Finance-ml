"""
Transaction categorization model.

Loads a trained TF-IDF + Logistic Regression model and provides a function,
predict_category(), that takes a transaction description and returns the
predicted category along with the model's confidence.

Usage:
    from categorize import predict_category, get_categories

    result = predict_category("STARBUCKS STORE #221")
    # {"category": "Dining", "confidence": 0.91}

    get_categories()
    # ["Dining", "Entertainment", "Fees", ...]
"""

import os

import joblib

# Resolve the model path relative to THIS FILE's location, not the current
# working directory. This is the fix for "works on my machine only" — the
# old version assumed the program was always run from a specific folder,
# which breaks the moment the backend imports this from somewhere else.
_MODEL_FILENAME = "categorizer.joblib"
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _MODEL_FILENAME)

_vectorizer = None
_model = None
_categories = None


def _load_model(model_path: str = _MODEL_PATH) -> None:
    """Load the trained model into module-level state. Raises a clear error
    if the model file is missing, instead of failing with a cryptic one."""
    global _vectorizer, _model, _categories

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Categorization model not found at: {model_path}\n"
            f"Make sure categorizer.joblib is in the same folder as categorize.py, "
            f"or pass an explicit path to _load_model()."
        )

    artifacts = joblib.load(model_path)
    _vectorizer = artifacts["vectorizer"]
    _model = artifacts["model"]
    _categories = sorted(_model.classes_.tolist())


# Load the model once, when this module is first imported.
_load_model()


def get_categories() -> list[str]:
    """Return the list of categories this model can predict."""
    return list(_categories)


def predict_category(text: str) -> dict:
    """
    Predict the spending category for a transaction description.

    Returns a dict: {"category": str, "confidence": float}
    confidence is the model's probability for the predicted category, 0-1.

    Raises ValueError if `text` is empty, whitespace-only, or not a string —
    callers (like the backend) should catch this and handle it as a bad
    request rather than letting it crash further down the pipeline.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "Transaction description must be a non-empty string. "
            f"Got: {text!r}"
        )

    vec = _vectorizer.transform([text])
    probabilities = _model.predict_proba(vec)[0]
    best_index = probabilities.argmax()

    return {
        # str(...) matters here: sklearn's .classes_ returns numpy string
        # types, which look fine in Python but aren't valid JSON on their
        # own — the backend will be serializing this dict directly.
        "category": str(_model.classes_[best_index]),
        "confidence": round(float(probabilities[best_index]), 4),
    }
