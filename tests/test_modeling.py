import pandas as pd

from src.config import TEST_PROCESSED_PATH
from src.modeling.predict import predict_default_probability


def test_prediction_output():
    if not TEST_PROCESSED_PATH.exists():
        return

    results = predict_default_probability()
    assert isinstance(results, pd.DataFrame)
    assert "Predicted_Default_Prob" in results.columns
    assert "Predicted_Status" in results.columns
    assert "Model" in results.columns
    assert "Model_Version" in results.columns

    probs = results["Predicted_Default_Prob"]
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert set(results["Predicted_Status"].unique()).issubset({0, 1})
