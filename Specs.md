# Minimal Local MLOps Learning Project Prompt

Generate a minimal, end-to-end local MLOps learning project designed to run on a standard laptop. Use Python 3.11, pandas, scikit-learn, FastAPI, and Docker Compose. The goal is to clearly demonstrate the full workflow from data loading through preprocessing, model training, artifact persistence, and serving via an API.

## Dataset:
- Use the Titanic survival dataset in CSV form, matching the Kaggle `train.csv` schema.
- Assume features: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`
- Target: `Survived`
- Either include a small embedded sample CSV or provide clear instructions for mounting the dataset locally.

## Model:
- Use `RandomForestClassifier` from scikit-learn.
- Keep hyperparameters simple and explicit.

## Training requirements:
- Implement a training script (`train.py`) that:
- Loads the CSV
- Performs basic preprocessing:
- Numeric: imputation (e.g., median)
- Categorical: imputation + one-hot encoding
- Use `ColumnTransformer` and `Pipeline`
- Splits data into train/validation sets
- Trains the model
- Outputs evaluation metrics: accuracy and ROC AUC
- Saves the full preprocessing + model pipeline using `joblib` to a `/models` directory

## Inference API requirements:
- Implement a FastAPI app (`predict_api.py`) that:
- Loads the saved pipeline artifact at startup
- Exposes a `/predict` POST endpoint:
- Accepts JSON input matching the feature schema
- Supports single and batch predictions
- Returns predicted class and probability
- Automatically exposes Swagger UI at `/docs` (FastAPI default)
- Uses Pydantic models for request/response schemas

## Architecture:
- Exactly two containers:
1. `trainer` (runs training script)
2. `api` (serves predictions)
- Use Docker Compose to orchestrate both services
- Use a shared volume for model artifacts (`/models`)
- The API container must depend on the trained model artifact existing

## Project structure:
- Provide a clean, minimal directory layout including:
- `src/` (training + API code)
- `data/` (dataset or placeholder)
- `models/` (artifacts)
- `docker/` (Dockerfiles)
- root files: `docker-compose.yml`, `requirements.txt`, `.env` (if needed), `README.md`

## Docker requirements:
- One Dockerfile for training
- One Dockerfile for API
- Keep images lightweight (e.g., python:3.11-slim)
- Avoid unnecessary dependencies

## Compose requirements:
- Define both services
- Mount volumes for data and models
- Expose API port (e.g., 8000)
- Ensure correct startup order or clear instructions

## Output requirements:
- Generate all files fully (no placeholders)
- Include:
- `requirements.txt`
- `train.py`
- `predict_api.py`
- Dockerfiles
- `docker-compose.yml`
- sample JSON request payloads
- example `curl` commands
- README with exact step-by-step commands:
1. prepare dataset
2. run training
3. start API
4. test `/predict`
5. access Swagger UI at `/docs`

## Constraints:
- Keep everything minimal and easy to understand
- Do not introduce additional services (no databases, no MLflow, no orchestration frameworks)
- Do not over-engineer abstractions
- All assumptions (paths, ports, file names) must be explicit
- Code should be runnable without modification on a typical laptop with Docker installed

## Goal:
Produce a small, complete, and transparent project that teaches the full ML lifecycle with clear, inspectable components.