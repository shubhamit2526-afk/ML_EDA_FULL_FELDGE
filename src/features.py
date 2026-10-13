import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    DROP_FEATURES,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    PROCESSED_DATA_DIR,
    TARGET_COL,
    TEST_PROCESSED_PATH,
    TRAIN_PROCESSED_PATH,
)
from src.dataset import load_raw_data
from src.logging_config import get_logger

logger = get_logger(__name__)


def execute_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df_fe = df.copy()

    for col in ["property_value", "dtir1", "income", "LTV"]:
        flag = f"{col}_isna"
        df_fe[flag] = df_fe[col].isnull().astype(int) if col in df_fe else 0

    income_monthly = (df_fe["income"] / 12.0).replace(0, np.nan) if "income" in df_fe else np.nan
    term_months = df_fe["term"].replace(0, np.nan) if "term" in df_fe else np.nan

    if "loan_amount" in df_fe and "term" in df_fe:
        df_fe["Est_Monthly_Payment"] = df_fe["loan_amount"] / term_months
        if "income" in df_fe:
            df_fe["Payment_to_Income"] = df_fe["Est_Monthly_Payment"] / income_monthly
            df_fe["Loan_to_Income"] = df_fe["loan_amount"] / (df_fe["income"] + 1.0)

    if "loan_amount" in df_fe and "property_value" in df_fe:
        df_fe["LTV_calculated"] = (df_fe["loan_amount"] / (df_fe["property_value"] + 1.0)) * 100.0

    if "dtir1" in df_fe and "LTV" in df_fe:
        df_fe["DTI_x_LTV"] = df_fe["dtir1"] * df_fe["LTV"]

    if "age" in df_fe:
        age_map = {
            "<25": 1,
            "25-34": 2,
            "35-44": 3,
            "45-54": 4,
            "55-64": 5,
            "65-74": 6,
            ">74": 7,
        }
        df_fe["age_ordinal"] = df_fe["age"].map(age_map).fillna(3)

    if "total_units" in df_fe:
        units_map = {"1U": 1, "2U": 2, "3U": 3, "4U": 4}
        df_fe["total_units_ordinal"] = df_fe["total_units"].map(units_map).fillna(1)

    return df_fe


def prepare_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering and remove leakage/unpredictive columns."""
    df_fe = execute_feature_engineering(df)
    cols_to_drop = [c for c in DROP_FEATURES if c in df_fe.columns]
    return df_fe.drop(columns=cols_to_drop)


def build_preprocessing_pipeline(num_cols, cat_cols, bin_cols):
    num_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    cat_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    bin_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
            ("bin", bin_transformer, bin_cols),
        ]
    )


def create_processed_datasets():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw dataset")
    df_raw = load_raw_data()

    logger.info("Executing feature engineering")
    df_clean = prepare_model_input(df_raw)

    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]

    num_cols = [c for c in NUMERICAL_FEATURES if c in X.columns]
    bin_cols = [c for c in BINARY_FEATURES if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols, bin_cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    logger.info("Fitting preprocessing pipeline on training data")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    cat_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(cat_names) + bin_cols

    df_train = pd.DataFrame(X_train_proc, columns=feature_names, index=X_train.index)
    df_train[TARGET_COL] = y_train
    df_test = pd.DataFrame(X_test_proc, columns=feature_names, index=X_test.index)
    df_test[TARGET_COL] = y_test

    df_train.sort_index().to_csv(TRAIN_PROCESSED_PATH, index=False)
    df_test.sort_index().to_csv(TEST_PROCESSED_PATH, index=False)

    # Standalone latest preprocessor; each model version also gets a copy during training.
    latest_preprocessor = MODELS_DIR / "latest_preprocessor.joblib"
    joblib.dump(preprocessor, latest_preprocessor)

    logger.info(
        "Processed datasets saved: train=%s test=%s features=%s",
        TRAIN_PROCESSED_PATH,
        TEST_PROCESSED_PATH,
        len(feature_names),
    )
    return preprocessor


if __name__ == "__main__":
    create_processed_datasets()
