"""
predict_api.py — Multi-model Titanic Survival Prediction API
=============================================================
This module runs a web server that discovers all trained model pipelines at
startup and serves predictions from whichever model the caller requests.

How it fits into the project:
  1. Each trainer (models/<name>/train.py) saves a pipeline.joblib into its own
     artifacts/ directory.
  2. Docker Compose mounts those directories into this container at /models/<name>/.
  3. On startup, this API scans /models/ for every pipeline.joblib it can find
     and loads them all into memory, keyed by their folder name.
  4. Callers POST passenger data to /predict?model=<name> and get back a
     survived flag plus a probability for each passenger.
  5. GET /models lists the names of every model currently available.

The contract every trainer must honour:
  Each pipeline.joblib must be a scikit-learn–compatible object that exposes
  .predict() and .predict_proba().  Both the Random Forest and Neural Net
  trainers satisfy this automatically because they wrap their models in a
  sklearn Pipeline.  If you add a non-sklearn model (e.g. TensorFlow), write a
  thin wrapper class with those two methods and joblib-dump the wrapper.

Interactive docs: http://localhost:8000/docs
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Root directory that contains one sub-folder per trained model.
# Docker Compose mounts ./models/<name>/artifacts/ at /models/<name>/ inside
# this container, so the API sees /models/random_forest/pipeline.joblib, etc.
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")

# Name of the model to use when the caller does not specify one.
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "random_forest")

# The exact set of columns every pipeline expects, in training order.
FEATURE_COLUMNS = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]

# Dict that maps model name → loaded pipeline object.  Populated at startup.
loaded_pipelines: dict = {}


# ---------------------------------------------------------------------------
# Startup / shutdown lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Scan MODELS_DIR at startup and load every pipeline.joblib found.

    Each subfolder of MODELS_DIR that contains a pipeline.joblib is registered
    under its folder name.  The API fails fast if no models are found so the
    problem is obvious immediately rather than at prediction time.
    """
    if not os.path.isdir(MODELS_DIR):
        raise RuntimeError(
            f"Models directory not found: {MODELS_DIR}. "
            "Check that the Docker volume mounts are correct in docker-compose.yml."
        )

    # Walk one level of subdirectories; each one may contain a pipeline.joblib.
    for model_name in sorted(os.listdir(MODELS_DIR)):
        artifact_path = os.path.join(MODELS_DIR, model_name, "pipeline.joblib")
        if os.path.isfile(artifact_path):
            loaded_pipelines[model_name] = joblib.load(artifact_path)
            print(f"Loaded model '{model_name}' from {artifact_path}")

    if not loaded_pipelines:
        raise RuntimeError(
            f"No pipeline.joblib files found under {MODELS_DIR}. "
            "Run the trainer containers before starting the API."
        )

    yield  # Serve requests until shutdown.

    loaded_pipelines.clear()


# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Titanic Survival Predictor",
    description=(
        "Predicts Titanic passenger survival using one of several trained models. "
        "Call GET /models to see what is available, then POST /predict?model=<name>."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request and response schemas (shared across all models)
# ---------------------------------------------------------------------------

class PassengerFeatures(BaseModel):
    """
    The details we need about one passenger to make a prediction.

    Age and Embarked are optional — if omitted the preprocessing pipeline
    fills them in automatically using values learned during training.
    """
    Pclass: int = Field(..., ge=1, le=3, description="Passenger class: 1, 2, or 3")
    Sex: str = Field(..., description="Passenger sex: 'male' or 'female'")
    Age: Optional[float] = Field(None, ge=0.0, description="Age in years (null = imputed)")
    SibSp: int = Field(..., ge=0, description="Number of siblings or spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents or children aboard")
    Fare: float = Field(..., ge=0.0, description="Ticket fare in British pounds")
    Embarked: Optional[str] = Field(None, description="Port of embarkation: C, Q, or S")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Pclass": 1,
                    "Sex": "female",
                    "Age": 29.0,
                    "SibSp": 0,
                    "Parch": 0,
                    "Fare": 211.34,
                    "Embarked": "S",
                }
            ]
        }
    }


class PredictionResult(BaseModel):
    """The model's verdict for a single passenger."""
    # 1 = the model predicts this passenger survived; 0 = did not survive.
    survived: int = Field(..., description="1 = survived, 0 = did not survive")
    # Confidence between 0 and 1.  Values above 0.5 led to survived=1.
    survival_probability: float = Field(..., description="Model confidence the passenger survived (0–1)")


class PredictionRequest(BaseModel):
    """One or more passengers to predict in a single API call."""
    passengers: list[PassengerFeatures] = Field(
        ..., min_length=1, description="One or more passengers to predict"
    )


class PredictionResponse(BaseModel):
    """One PredictionResult per passenger, in the same order as the request."""
    model_used: str = Field(..., description="Name of the model that produced these predictions")
    predictions: list[PredictionResult]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/models", summary="List available models")
def list_models() -> dict:
    """
    Return the names of all models currently loaded and ready to serve
    predictions.  Use one of these names as the ?model= query parameter on
    POST /predict.
    """
    return {"models": list(loaded_pipelines.keys())}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    model: str = Query(
        DEFAULT_MODEL,
        description="Name of the model to use.  Call GET /models to see what is available.",
    ),
) -> PredictionResponse:
    """
    Accept one or more passengers and return a survival prediction for each.

    The ?model= query parameter selects which trained model to use.
    If omitted, the DEFAULT_MODEL environment variable is used (random_forest).

    Steps:
      1. Look up the requested model in the loaded_pipelines dictionary.
      2. Convert the incoming JSON into a pandas DataFrame.
      3. Select only the columns the pipeline was trained on.
      4. Ask the pipeline for predicted classes and survival probabilities.
      5. Return one result per passenger.
    """
    if model not in loaded_pipelines:
        available = list(loaded_pipelines.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' is not loaded.  Available models: {available}",
        )

    active_pipeline = loaded_pipelines[model]

    # Turn the list of passenger objects into a table with one row per passenger.
    passenger_table = pd.DataFrame([p.model_dump() for p in request.passengers])

    # Keep only the features the pipeline was trained on, in the correct order.
    passenger_table = passenger_table[FEATURE_COLUMNS]

    # predict()       → array of 0s and 1s
    # predict_proba() → array of [prob_class_0, prob_class_1] per row;
    #                   [:, 1] selects the "survived" probability column.
    predicted_classes = active_pipeline.predict(passenger_table)
    predicted_survival_probability = active_pipeline.predict_proba(passenger_table)[:, 1]

    results = [
        PredictionResult(
            survived=int(predicted_class),
            survival_probability=round(float(probability), 4),
        )
        for predicted_class, probability in zip(predicted_classes, predicted_survival_probability)
    ]

    return PredictionResponse(model_used=model, predictions=results)


@app.get("/health", summary="Health check")
def health() -> dict:
    """
    Liveness and readiness check.

    Returns the names of all loaded models.  If the list is empty the service
    is degraded.  Useful for monitoring and Docker health-check directives.
    """
    loaded_model_names = list(loaded_pipelines.keys())
    is_ready = len(loaded_model_names) > 0
    return {
        "status": "ok" if is_ready else "degraded",
        "models_loaded": loaded_model_names,
    }
