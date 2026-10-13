import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import (
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    DEFAULT_VERSION,
    LATEST_MODEL_FILE,
    MODEL_VERSIONS_DIR,
    TARGET_COL,
    TEST_PROCESSED_PATH,
)
from src.features import prepare_model_input
from src.logging_config import get_logger

logger = get_logger(__name__)


def resolve_model(model_name=DEFAULT_MODEL, version=DEFAULT_VERSION):
    if LATEST_MODEL_FILE.exists() and version == DEFAULT_VERSION:
        latest = json.loads(LATEST_MODEL_FILE.read_text(encoding="utf-8"))
        version = latest["version"]

    version_dir = MODEL_VERSIONS_DIR / version
    model_path = version_dir / f"{model_name}.joblib"
    preprocessor_path = version_dir / "preprocessor.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {preprocessor_path}")

    return version, joblib.load(model_path), joblib.load(preprocessor_path)


def predict_default_probability(
    data_path: Path = TEST_PROCESSED_PATH,
    model_name=DEFAULT_MODEL,
    version=DEFAULT_VERSION,
):
    """Batch prediction for already-processed test data."""
    version, model, _ = resolve_model(model_name, version)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)
    X = df.drop(columns=[TARGET_COL]) if TARGET_COL in df else df
    probabilities = model.predict_proba(X.to_numpy())[:, 1]
    predictions = (probabilities >= DEFAULT_THRESHOLD).astype(int)

    result = df.copy()
    result["Predicted_Default_Prob"] = probabilities
    result["Predicted_Status"] = predictions
    result["Model"] = model_name
    result["Model_Version"] = version
    logger.info("Batch prediction completed for %d rows using %s/%s", len(result), model_name, version)
    return result


def predict_single(application_dict, model_name=DEFAULT_MODEL, version=DEFAULT_VERSION):
    version, model, preprocessor = resolve_model(model_name, version)

    raw = pd.DataFrame([application_dict])
    X_raw = prepare_model_input(raw)

    X_processed = preprocessor.transform(X_raw)
    probability = float(model.predict_proba(X_processed)[0, 1])
    prediction = int(probability >= DEFAULT_THRESHOLD)

    logger.info("Single prediction completed using %s/%s probability=%.4f", model_name, version, probability)

    return {
        "default_probability": round(probability, 4),
        "predicted_status": prediction,
        "underwriting_recommendation": ("REJECT / HIGH RISK" if prediction == 1 else "APPROVE / LOW RISK"),
        "model": model_name,
        "model_version": version,
    }


if __name__ == "__main__":
    result = predict_default_probability()
    print(result[["Predicted_Default_Prob", "Predicted_Status", "Model", "Model_Version"]].head(10))
