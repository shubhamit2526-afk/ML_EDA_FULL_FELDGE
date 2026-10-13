from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "Loan_Default.csv"
TRAIN_PROCESSED_PATH = PROCESSED_DATA_DIR / "train_processed.csv"
TEST_PROCESSED_PATH = PROCESSED_DATA_DIR / "test_processed.csv"

MODELS_DIR = PROJ_ROOT / "models"
MODEL_VERSIONS_DIR = MODELS_DIR / "versions"
LATEST_MODEL_FILE = MODELS_DIR / "latest.json"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = PROJ_ROOT / "logs"

TARGET_COL = "Status"

LEAKAGE_COLS = ["rate_of_interest", "Interest_rate_spread", "Upfront_charges"]

ZERO_VAR_COLS = ["ID", "year", "construction_type", "Secured_by", "Security_Type", "open_credit"]

UNPREDICTIVE_COLS = [
    "Credit_Score",
    "term",
    "Gender",
    "Region",
    "approv_in_adv",
    "loan_limit",
    "business_or_commercial",
    "credit_type",
    "Credit_Worthiness",
    "interest_only",
]

DROP_FEATURES = list(dict.fromkeys(LEAKAGE_COLS + ZERO_VAR_COLS + UNPREDICTIVE_COLS))

NUMERICAL_FEATURES = [
    "property_value",
    "dtir1",
    "LTV",
    "income",
    "loan_amount",
    "LTV_calculated",
    "Payment_to_Income",
    "Loan_to_Income",
    "DTI_x_LTV",
]

BINARY_FEATURES = ["property_value_isna", "dtir1_isna", "income_isna", "LTV_isna", "age_ordinal", "total_units_ordinal"]

CATEGORICAL_FEATURES = [
    "loan_type",
    "loan_purpose",
    "occupancy_type",
    "submission_of_application",
    "Neg_ammortization",
    "lump_sum_payment",
    "co-applicant_credit_type",
]

DEFAULT_MODEL = "random_forest"
DEFAULT_VERSION = "v1"
DEFAULT_THRESHOLD = 0.50
