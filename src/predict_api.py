"""
predict_api.py — Titanic Survival Prediction API
=================================================
This module runs a web server that accepts passenger details and returns a
survival prediction from the trained machine-learning pipeline.

How it fits into the project:
  1. train.py runs first and saves a fitted pipeline to models/pipeline.joblib.
  2. This API loads that file once at startup.
  3. Callers POST passenger data to /predict and receive a survived flag plus
     a probability score for each passenger.

The server is built with FastAPI, which automatically generates interactive
documentation at /docs — no extra tooling required.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Where the trained pipeline was saved by train.py.
# Overridable via the MODEL_PATH environment variable so Docker Compose can
# point both containers to the same shared directory.
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/pipeline.joblib")

# The exact set of columns the pipeline expects, in the order it was trained on.
# Keeping this list here makes it easy to spot if the API and trainer ever drift
# out of sync.
FEATURE_COLUMNS = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]

# Module-level variable that holds the loaded pipeline.
# It is populated during startup (see `lifespan` below) and stays in memory
# for the lifetime of the server process.
pipeline = None


# ---------------------------------------------------------------------------
# Startup / shutdown lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI calls this function when the server starts and when it shuts down.

    On startup  → load the pipeline from disk into memory once.
    On shutdown → release the reference so Python can free the memory.

    Loading the pipeline here (rather than inside each request) means the
    file is only read from disk once, keeping prediction requests fast.
    """
    global pipeline

    # Fail loudly at startup if the model file is missing — better to crash
    # immediately than to accept requests and fail later.
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run the trainer container first."
        )

    pipeline = joblib.load(MODEL_PATH)
    print(f"Pipeline loaded from {MODEL_PATH}")

    yield  # Server is running; handle requests until shutdown is requested.

    pipeline = None  # Clean up on shutdown.


# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Titanic Survival Predictor",
    description=(
        "Predicts survival probability for Titanic passengers. "
        "Accepts single or batch requests."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------------------------

class PassengerFeatures(BaseModel):
    """
    The details we need about one passenger to make a prediction.

    Age and Embarked are optional — if omitted the preprocessing pipeline
    fills them in automatically using values learned during training
    (median age and most-common port).
    """
    Pclass: int = Field(..., ge=1, le=3, description="Passenger class: 1, 2, or 3")
    Sex: str = Field(..., description="Passenger sex: 'male' or 'female'")
    Age: Optional[float] = Field(None, ge=0.0, description="Age in years (null = imputed)")
    SibSp: int = Field(..., ge=0, description="Number of siblings or spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents or children aboard")
    Fare: float = Field(..., ge=0.0, description="Ticket fare in British pounds")
    Embarked: Optional[str] = Field(None, description="Port of embarkation: C, Q, or S")

    # Example payload shown in the auto-generated /docs UI.
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
    survived: int = Field(..., description="Predicted class: 1 = survived, 0 = did not survive")
    # A number between 0 and 1. Values above 0.5 led to survived=1.
    # Example: 0.93 means the model is 93 % confident this passenger survived.
    survival_probability: float = Field(..., description="Model confidence that the passenger survived")


class PredictionRequest(BaseModel):
    """Wrapper that holds one or more passengers in a single API call (batch support)."""
    passengers: list[PassengerFeatures] = Field(
        ..., min_length=1, description="One or more passengers to predict"
    )


class PredictionResponse(BaseModel):
    """The API response: one PredictionResult per passenger, in the same order."""
    predictions: list[PredictionResult]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Accept one or more passengers and return a survival prediction for each.

    Steps:
      1. Convert the incoming JSON into a pandas DataFrame the pipeline understands.
      2. Select only the columns the pipeline was trained on (in the right order).
      3. Ask the pipeline for predicted classes (0 or 1) and survival probabilities.
      4. Pair each prediction with its probability and return the list.
    """
    # Guard against the unlikely case that the model failed to load at startup.
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # Turn the list of passenger objects into a table (DataFrame) with one row
    # per passenger and one column per feature.
    passenger_table = pd.DataFrame([p.model_dump() for p in request.passengers])

    # Keep only the features the model was trained on, in the correct column order.
    passenger_table = passenger_table[FEATURE_COLUMNS]

    # pipeline.predict()       → array of 0s and 1s (did not survive / survived)
    # pipeline.predict_proba() → array of [prob_class_0, prob_class_1] per row;
    #                            [:, 1] selects only the "survived" probability column.
    predicted_classes = pipeline.predict(passenger_table)
    predicted_survival_probability = pipeline.predict_proba(passenger_table)[:, 1]

    # Zip the two arrays together so each passenger gets their own result object.
    results = [
        PredictionResult(
            survived=int(predicted_class),
            survival_probability=round(float(probability), 4),
        )
        for predicted_class, probability in zip(predicted_classes, predicted_survival_probability)
    ]
    return PredictionResponse(predictions=results)


@app.get("/health", summary="Health check")
def health() -> dict:
    """
    Simple liveness/readiness check.

    Returns 'ok' once the pipeline is loaded, 'degraded' if it is not.
    Useful for monitoring tools and Docker health-check directives.
    """
    is_ready = pipeline is not None
    return {"status": "ok" if is_ready else "degraded", "model_loaded": is_ready}
