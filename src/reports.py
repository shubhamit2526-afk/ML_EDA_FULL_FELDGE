import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import DEFAULT_VERSION, MODEL_VERSIONS_DIR, REPORTS_DIR, TARGET_COL, TEST_PROCESSED_PATH
from src.logging_config import get_logger

logger = get_logger(__name__)


def generate_model_report(model_name, version=DEFAULT_VERSION):
    version_dir = MODEL_VERSIONS_DIR / version
    model = joblib.load(version_dir / f"{model_name}.joblib")
    test_df = pd.read_csv(TEST_PROCESSED_PATH)

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.50).astype(int)
    cm = confusion_matrix(y_test, preds)

    metrics = {
        "ROC_AUC": round(float(roc_auc_score(y_test, probs)), 4),
        "PR_AUC": round(float(average_precision_score(y_test, probs)), 4),
        "Accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "Precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "F1_Score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "Confusion_Matrix": {
            "True_Negative": int(cm[0, 0]),
            "False_Positive": int(cm[0, 1]),
            "False_Negative": int(cm[1, 0]),
            "True_Positive": int(cm[1, 1]),
        },
    }

    report_dir = REPORTS_DIR / model_name
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fpr, tpr, _ = roc_curve(y_test, probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={metrics['ROC_AUC']:.4f})")
    ax.plot([0, 1], [0, 1], "--")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "roc_curve.png", dpi=200)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_test, probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label=f"{model_name} (PR-AUC={metrics['PR_AUC']:.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "pr_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cbar=False,
        ax=ax,
        xticklabels=["Non-Default", "Default"],
        yticklabels=["Non-Default", "Default"],
    )
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix")
    fig.tight_layout()
    fig.savefig(figure_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    if hasattr(model, "feature_importances_"):
        fi = (
            pd.DataFrame(
                {
                    "Feature": X_test.columns,
                    "Importance": model.feature_importances_,
                }
            )
            .sort_values("Importance", ascending=False)
            .head(15)
        )
        fi.to_csv(report_dir / "feature_importance.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 7))
        ordered = fi.sort_values("Importance")
        ax.barh(ordered["Feature"], ordered["Importance"])
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title(f"Top 15 Feature Importances - {model_name}")
        fig.tight_layout()
        fig.savefig(figure_dir / "feature_importance.png", dpi=200)
        plt.close(fig)

    return metrics


def generate_all_reports(version=DEFAULT_VERSION):
    models = ["logistic_regression", "random_forest", "xgboost"]
    comparison = {}
    for name in models:
        comparison[name] = generate_model_report(name, version)

    comparison_df = pd.DataFrame(comparison).T
    comparison_df.drop(columns=["Confusion_Matrix"]).to_csv(REPORTS_DIR / "model_comparison.csv")
    (REPORTS_DIR / "model_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    logger.info("Generated reports for %d models", len(models))


if __name__ == "__main__":
    generate_all_reports()
