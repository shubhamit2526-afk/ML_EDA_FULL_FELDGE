from src.config import LATEST_MODEL_FILE, MODEL_VERSIONS_DIR
from src.modeling.train import train_models


def test_training_creates_versioned_artifacts():
    metadata = train_models("test-version")
    assert metadata["version"] == "test-version"
    version_dir = MODEL_VERSIONS_DIR / "test-version"
    assert (version_dir / "logistic_regression.joblib").exists()
    assert (version_dir / "random_forest.joblib").exists()
    assert (version_dir / "xgboost.joblib").exists()
    assert (version_dir / "preprocessor.joblib").exists()
    assert (version_dir / "metadata.json").exists()
    assert LATEST_MODEL_FILE.exists()
