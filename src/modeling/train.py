import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    DEFAULT_VERSION,
    LATEST_MODEL_FILE,
    MODEL_VERSIONS_DIR,
    TARGET_COL,
    TEST_PROCESSED_PATH,
    TRAIN_PROCESSED_PATH,
)
from src.features import create_processed_datasets
from src.logging_config import get_logger

logger = get_logger(__name__)


def _metrics(y_true, probabilities, threshold=0.50):
    predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, predictions)
    return {
        "ROC_AUC": round(float(roc_auc_score(y_true, probabilities)), 4),
        "PR_AUC": round(float(average_precision_score(y_true, probabilities)), 4),
        "Accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "Precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "F1_Score": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "Confusion_Matrix": {
            "True_Negative": int(cm[0, 0]),
            "False_Positive": int(cm[0, 1]),
            "False_Negative": int(cm[1, 0]),
            "True_Positive": int(cm[1, 1]),
        },
    }


def train_models(version: str = DEFAULT_VERSION):
    """Train LR, RF and XGBoost and save all artifacts under one version."""
    version_dir = MODEL_VERSIONS_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    if not TRAIN_PROCESSED_PATH.exists() or not TEST_PROCESSED_PATH.exists():
        preprocessor = create_processed_datasets()
    else:
        preprocessor = joblib.load(Path(__file__).resolve().parents[2] / "models" / "latest_preprocessor.joblib")

    train_df = pd.read_csv(TRAIN_PROCESSED_PATH)
    test_df = pd.read_csv(TEST_PROCESSED_PATH)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    from xgboost import XGBClassifier

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }

    metadata = {
        "version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": __import__("sklearn").__version__,
        "models": {},
    }

    joblib.dump(preprocessor, version_dir / "preprocessor.joblib")

    for name, model in models.items():
        logger.info("Training %s", name)
        model.fit(X_train.to_numpy(), y_train.to_numpy())
        probabilities = model.predict_proba(X_test.to_numpy())[:, 1]
        metrics = _metrics(y_test, probabilities)

        artifact = version_dir / f"{name}.joblib"
        joblib.dump(model, artifact)

        metadata["models"][name] = {
            "artifact": artifact.name,
            "metrics": metrics,
            "threshold": 0.50,
        }
        logger.info(
            "%s ROC-AUC=%.4f PR-AUC=%.4f F1=%.4f", name, metrics["ROC_AUC"], metrics["PR_AUC"], metrics["F1_Score"]
        )

    with open(version_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(LATEST_MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": version,
                "default_model": "random_forest",
                "metadata": str(version_dir / "metadata.json"),
            },
            f,
            indent=2,
        )

    logger.info("Model version %s saved to %s", version, version_dir)
    return metadata


if __name__ == "__main__":
    train_models()
