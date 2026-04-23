# MLOps Training Project

A minimal, end-to-end local MLOps project demonstrating the full ML lifecycle:

```
data loading → preprocessing → training → artifact persistence → serving via API
```

Two models are included side by side so you can compare approaches:
- **Random Forest** — an ensemble of decision trees (no feature scaling required)
- **Neural Network** — a Multi-Layer Perceptron (requires normalised inputs)

Both are served by a single API. You choose which model to query per request.

## Project Structure

```
MlOpsTraining/
├── data/
│   └── train.csv                        # 90-row Titanic sample (or drop in the full Kaggle CSV)
├── models/
│   ├── random_forest/
│   │   ├── train.py                     # Training script
│   │   ├── Dockerfile                   # Trainer container image
│   │   ├── requirements.txt             # sklearn + pandas only
│   │   └── artifacts/                   # pipeline.joblib written here after training
│   └── neural_net/
│       ├── train.py                     # Training script
│       ├── Dockerfile                   # Trainer container image
│       ├── requirements.txt             # sklearn + pandas only
│       └── artifacts/                   # pipeline.joblib written here after training
├── api/
│   ├── predict_api.py                   # FastAPI service — loads all models at startup
│   ├── Dockerfile                       # API container image
│   └── requirements.txt                 # fastapi + sklearn + pandas
├── docs/
│   └── random-forest.md                 # Plain-English explanation of Random Forest
├── docker-compose.yml                   # Orchestrates trainers → api
├── reset.ps1                            # Stop, clean artifacts, rebuild, restart
└── README.md
```

**To add a new model:** create a `models/<name>/` folder following the same layout, add a
`trainer-<name>` service in `docker-compose.yml`, and mount its `artifacts/` into the API
service. No other files change.

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

### 2. Train All Models and Start the API

```powershell
docker compose up --build
```

Docker builds both trainer images, runs them in parallel, and starts the API once both
exit successfully. The trained pipelines are written to each model's `artifacts/` folder
on the host.

Expected training output (one block per model):
```
trainer-random-forest-1  | Loading data from /data/train.csv
trainer-random-forest-1  | Loaded 90 rows
trainer-random-forest-1  | Train size: 72, Validation size: 18
trainer-random-forest-1  | Validation Accuracy:                    0.XXXX
trainer-random-forest-1  | Correct Ranking Probability (ROC AUC):  0.XXXX
trainer-random-forest-1  | Pipeline saved to /artifacts/pipeline.joblib
trainer-random-forest-1 exited with code 0

trainer-neural-net-1     | ...same output...
trainer-neural-net-1 exited with code 0
```

The API is ready when you see:
```
api-1  | Loaded model 'neural_net' from /models/neural_net/pipeline.joblib
api-1  | Loaded model 'random_forest' from /models/random_forest/pipeline.joblib
api-1  | INFO:     Application startup complete.
api-1  | INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Check Available Models

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/models"
```

Response:
```json
{ "models": ["neural_net", "random_forest"] }
```

### 4. Test the `/predict` Endpoint

Use the `?model=` query parameter to choose which model answers. If omitted, `random_forest`
is used by default.

**Single prediction — first-class female, using the Random Forest:**

```powershell
$body = '{"passengers":[{"Pclass":1,"Sex":"female","Age":29.0,"SibSp":0,"Parch":0,"Fare":211.34,"Embarked":"S"}]}'
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/predict?model=random_forest" `
  -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
```

**Same passenger, using the Neural Network:**

```powershell
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/predict?model=neural_net" `
  -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
```

Example response (both models return the same shape):
```json
{
  "model_used": "random_forest",
  "predictions": [
    {
      "survived": 1,
      "survival_probability": 0.9390
    }
  ]
}
```

---

**Batch prediction — two passengers:**

```powershell
$body = '{"passengers":[{"Pclass":3,"Sex":"male","Age":22.0,"SibSp":1,"Parch":0,"Fare":7.25,"Embarked":"S"},{"Pclass":1,"Sex":"female","Age":38.0,"SibSp":1,"Parch":0,"Fare":71.28,"Embarked":"C"}]}'
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/predict?model=random_forest" `
  -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
```

---

**Prediction with missing Age (imputed automatically by the pipeline):**

```powershell
$body = '{"passengers":[{"Pclass":2,"Sex":"male","Age":null,"SibSp":0,"Parch":0,"Fare":13.0,"Embarked":"S"}]}'
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/predict" `
  -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
```

### 5. Access Swagger UI

Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

The interactive Swagger UI lets you browse all endpoints, see the `?model=` parameter as a
text field, and send live test requests without installing any tools.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Docker Compose                                                     │
│                                                                     │
│  ┌──────────────────────────┐  artifacts/pipeline.joblib           │
│  │  trainer-random-forest   │ ──────────────────────────┐          │
│  │  (exits 0)               │                           │          │
│  └──────────────────────────┘                           ▼          │
│                                               ┌──────────────────┐ │
│  ┌──────────────────────────┐                 │       api        │ │
│  │  trainer-neural-net      │ ──────────────▶ │  predict_api.py  │ │
│  │  (exits 0)               │  artifacts/     │  :8000           │ │
│  └──────────────────────────┘  pipeline.joblib│  ?model=...      │ │
│                                               └──────────────────┘ │
│  Both trainers read ./data/train.csv (read-only mount)             │
└─────────────────────────────────────────────────────────────────────┘
```

| Component                   | Role                                                              |
|-----------------------------|-------------------------------------------------------------------|
| `trainer-random-forest`     | One-shot container: trains RF, saves pipeline to `artifacts/`    |
| `trainer-neural-net`        | One-shot container: trains MLP, saves pipeline to `artifacts/`   |
| `api`                       | Long-running container: loads all pipelines, serves predictions  |
| `models/<name>/artifacts/`  | Per-model output directory, visible on the host after training   |

## Key Design Decisions

- **One folder per model** — every file related to a model (code, Dockerfile, dependencies, output) lives under `models/<name>/`. Removing a model means deleting its folder and its two lines in `docker-compose.yml`.
- **Separate `requirements.txt` per container** — the neural net trainer does not install FastAPI; the API does not need to know about training utilities. Keeps images small and dependency trees simple.
- **Common pipeline interface** — every trainer saves a joblib-wrapped object with `.predict()` and `.predict_proba()`. The API never needs to know what algorithm is inside.
- **API auto-discovery** — the API scans `/models/*/pipeline.joblib` at startup. No hardcoded model list; adding a model only requires a new volume mount.
- **`?model=` query parameter** — visible as a text field in Swagger UI; defaults to `random_forest` if omitted.
- **`service_completed_successfully`** — the API only starts after all trainers exit with code 0.
- **Bind-mount volumes** (not named volumes) keep `artifacts/pipeline.joblib` visible on the host for inspection.
- **Pydantic v2 models** provide type-safe, auto-documented request/response schemas.

## Stopping Services

```powershell
docker compose down
```

## Resetting and Retraining

A convenience script stops containers, deletes all saved model artifacts, and rebuilds/restarts everything.

```powershell
.\reset.ps1
```

> Note the `.\` prefix. PowerShell requires it to run scripts in the current directory.

The script runs Docker Compose in detached mode so the terminal returns immediately and the
API continues running in the background. Use `docker compose down` to stop it.

