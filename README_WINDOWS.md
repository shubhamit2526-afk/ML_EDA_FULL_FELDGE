# Windows Setup and Workflow

## 1. Prerequisites

Install Python 3.10.x, Git and VS Code.

Do not install Rust for Ruff. Do not install GNU Make unless you specifically want to keep using the old Makefile.

## 2. Create the environment

PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again.

## 3. Run the pipeline

```powershell
python -m src.dataset
python -m src.features
python -m src.train
python -m src.predict
python -m src.reports
```

Or use Taskipy:

```powershell
task features
task train
task reports
task test
task lint
task format
task all
```

## 4. Flask prediction API

```powershell
python -m src.api
```

Health check:
GET http://127.0.0.1:5000/health

Prediction:
POST http://127.0.0.1:5000/predict?model=random_forest&version=v1

## 5. Artifacts

models/versions/v1/
    logistic_regression.joblib
    random_forest.joblib
    xgboost.joblib
    preprocessor.joblib
    metadata.json

reports/
    logistic_regression/
    random_forest/
    xgboost/
    model_comparison.csv
    model_comparison.json

logs/app.log
