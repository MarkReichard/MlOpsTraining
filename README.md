# MLOps Training Project

A minimal, end-to-end local MLOps project demonstrating the full ML lifecycle:

```
data loading → preprocessing → training → artifact persistence → serving via API
```

## Project Structure

```
MlOpsTraining/
├── data/
│   └── train.csv            # 90-row Titanic sample (or drop in the full Kaggle CSV)
├── docker/
│   ├── Dockerfile.api       # API container image
│   └── Dockerfile.trainer   # Training container image
├── models/                  # Saved pipeline artifacts (populated after training)
├── src/
│   ├── predict_api.py       # FastAPI prediction service
│   └── train.py             # Training script
├── docker-compose.yml       # Orchestrates trainer → api
├── requirements.txt         # Python dependencies (shared)
└── README.md
```

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with Docker Compose v2  
  (`docker compose version` should report v2.1 or later)
- No local Python installation required — everything runs inside containers

## Step-by-Step Instructions

### 1. Prepare the Dataset

A 90-row Titanic sample is already included at `data/train.csv`.

To use the full 891-row Kaggle dataset instead:
1. Download `train.csv` from the [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data)
2. Replace `data/train.csv` with the downloaded file

### 2. Run Training

```bash
docker compose up trainer
```

Docker builds the trainer image, runs `train.py`, and writes the fitted pipeline to
`models/pipeline.joblib` on the host via the bind-mount volume.

Expected output:
```
trainer-1  | Loading data from /data/train.csv
trainer-1  | Loaded 90 rows
trainer-1  | Train size: 72, Validation size: 18
trainer-1  | Validation Accuracy: 0.XXXX
trainer-1  | Validation ROC AUC:  0.XXXX
trainer-1  | Pipeline saved to /models/pipeline.joblib
trainer-1 exited with code 0
```

### 3. Start the Prediction API

```bash
docker compose up api
```

Because `depends_on: condition: service_completed_successfully` is set, the `api` service
automatically waits for the `trainer` to exit with code 0. You can also run both at once:

```bash
docker compose up
```

The API is ready when you see:
```
api-1  | INFO:     Application startup complete.
api-1  | INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4. Test the `/predict` Endpoint

**Single prediction — first-class female (likely survivor):**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "passengers": [
      {
        "Pclass": 1,
        "Sex": "female",
        "Age": 29.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 211.34,
        "Embarked": "S"
      }
    ]
  }'
```

Example response:
```json
{
  "predictions": [
    {
      "survived": 1,
      "survival_probability": 0.92
    }
  ]
}
```

---

**Batch prediction — two passengers:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "passengers": [
      {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
      },
      {
        "Pclass": 1,
        "Sex": "female",
        "Age": 38.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 71.28,
        "Embarked": "C"
      }
    ]
  }'
```

---

**Prediction with missing Age (imputed automatically by the pipeline):**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "passengers": [
      {
        "Pclass": 2,
        "Sex": "male",
        "Age": null,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 13.0,
        "Embarked": "S"
      }
    ]
  }'
```

### 5. Access Swagger UI

Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

The interactive Swagger UI lets you browse endpoints, inspect Pydantic schemas, and send
live test requests without installing any tools.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Docker Compose                                          │
│                                                          │
│  ┌───────────────┐   ./models/ bind-mount                │
│  │   trainer     │ ──────────────────────▶ pipeline.joblib│
│  │   train.py    │                              │        │
│  │   (exits 0)   │                              │        │
│  └───────────────┘                              ▼        │
│         │                          ┌───────────────────┐ │
│    ./data/train.csv                │       api         │ │
│                     depends_on ──▶ │  predict_api.py   │ │
│               (completed_success.) │  :8000            │ │
│                                    └───────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

| Component         | Role                                                        |
|-------------------|-------------------------------------------------------------|
| `trainer`         | One-shot container: trains, evaluates, saves pipeline       |
| `api`             | Long-running container: loads pipeline, serves predictions  |
| `./models/`       | Shared bind-mount directory visible on the host             |
| `ColumnTransformer + Pipeline` | Bundles preprocessing and classifier into one serializable artifact |

## Key Design Decisions

- **`ColumnTransformer`** handles numeric (median imputation) and categorical (mode imputation + one-hot encoding) features in a single step.
- **`Pipeline`** chains preprocessing and `RandomForestClassifier` into one object, preventing data-leakage during fit and simplifying serialization.
- **`joblib`** serializes the full pipeline to a single `.joblib` file — loaded once at API startup.
- **Bind-mount volumes** (not named volumes) keep `models/pipeline.joblib` visible on the host for inspection.
- **`service_completed_successfully`** ensures the API only starts after a successful training run.
- **Pydantic v2 models** provide type-safe, auto-documented request/response schemas with `null`-safe optional fields (`Age`, `Embarked`).

## Stopping Services

```bash
docker compose down
```

To remove the trained model and retrain from scratch:

```bash
rm models/pipeline.joblib
docker compose up
```
