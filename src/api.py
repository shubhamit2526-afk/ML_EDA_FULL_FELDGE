"""Flask API for the prediction endpoint only."""

from flask import Flask, jsonify, request

from src.logging_config import get_logger
from src.modeling.predict import predict_single

logger = get_logger(__name__)
app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    try:
        result = predict_single(
            payload,
            model_name=request.args.get("model", "random_forest"),
            version=request.args.get("version", "v1"),
        )
        return jsonify(result), 200
    except Exception:
        logger.exception("Prediction request failed")
        return jsonify({"error": "Prediction failed; check server logs"}), 500


if __name__ == "__main__":
    logger.info("Starting Flask prediction API on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
