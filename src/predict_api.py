import os
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/pipeline.joblib")

FEATURE_COLUMNS = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]

pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run the trainer container first."
        )
    pipeline = joblib.load(MODEL_PATH)
    print(f"Pipeline loaded from {MODEL_PATH}")
    yield
    pipeline = None


app = FastAPI(
    title="Titanic Survival Predictor",
    description=(
        "Predicts survival probability for Titanic passengers. "
        "Accepts single or batch requests."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PassengerFeatures(BaseModel):
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
    survived: int = Field(..., description="Predicted class: 1 = survived, 0 = did not survive")
    survival_probability: float = Field(..., description="Model confidence that the passenger survived")


class PredictionRequest(BaseModel):
    passengers: list[PassengerFeatures] = Field(
        ..., min_length=1, description="One or more passengers to predict"
    )


class PredictionResponse(BaseModel):
    predictions: list[PredictionResult]


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    df = pd.DataFrame([p.model_dump() for p in request.passengers])
    df = df[FEATURE_COLUMNS]

    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)[:, 1]

    results = [
        PredictionResult(survived=int(pred), survival_probability=round(float(prob), 4))
        for pred, prob in zip(predictions, probabilities)
    ]
    return PredictionResponse(predictions=results)


@app.get("/health", summary="Health check")
def health() -> dict:
    is_ready = pipeline is not None
    return {"status": "ok" if is_ready else "degraded", "model_loaded": is_ready}
