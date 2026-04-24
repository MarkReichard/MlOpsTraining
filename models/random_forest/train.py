"""
train.py — Random Forest training script (Titanic survival)
============================================================
Loads the Titanic CSV, preprocesses features, trains a RandomForestClassifier,
prints validation metrics, and saves the full preprocessing + model pipeline to
disk so the prediction API can load it without repeating any of this logic.

For a plain-English explanation of how RandomForestClassifier works and what it
is good for, see docs/random-forest.md.

Output: /artifacts/pipeline.joblib
  A single serialised object that contains both the preprocessing steps and the
  trained classifier.  The API loads this file at startup.
"""

import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Paths — overridable via environment variables so Docker Compose can inject
# the correct container paths without changing this file.
# ---------------------------------------------------------------------------
DATA_PATH = os.environ.get("DATA_PATH", "/data/train.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", "/artifacts/pipeline.joblib")

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["PassengerClass", "Age", "SiblingsOrSpouses", "ParentsOrChildren", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "PortOfEmbarkation"]
TARGET = "Survived"

# ---------------------------------------------------------------------------
# Hyperparameters — named constants instead of magic numbers.
# ---------------------------------------------------------------------------
RANDOM_STATE = 42   # Fixed seed → reproducible model every run.
TEST_SIZE = 0.2     # 20 % held back for validation; 80 % used to train.
N_ESTIMATORS = 100  # Number of trees in the forest.
MAX_DEPTH = 5       # Max levels per tree. Shallow trees generalise better on
                    # small datasets by avoiding memorising the training data.


def load_data(path: str) -> pd.DataFrame:
    """Load CSV and verify all required columns are present."""
    passenger_data = pd.read_csv(path)
    passenger_data = passenger_data.rename(columns={
        "Pclass": "PassengerClass",
        "SibSp": "SiblingsOrSpouses",
        "Parch": "ParentsOrChildren",
        "Embarked": "PortOfEmbarkation",
    })
    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [col for col in required_columns if col not in passenger_data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return passenger_data


def build_pipeline() -> Pipeline:
    """
    Construct the preprocessing + classifier pipeline.

    Preprocessing steps:
    - Numeric columns  : fill missing values with the column median.
    - Categorical columns: fill missing values with the most common value,
      then one-hot encode (converts e.g. "male"/"female" to 0/1 columns).

    Why a Pipeline?
    Bundling preprocessing and the classifier into one object ensures the same
    transformation is applied consistently during training and at prediction
    time, and lets the whole thing serialise to a single file with joblib.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
        )),
    ])


def main() -> None:
    print(f"Loading data from {DATA_PATH}")
    passenger_data = load_data(DATA_PATH)
    print(f"Loaded {len(passenger_data)} rows")

    features = passenger_data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    labels = passenger_data[TARGET]

    training_features, validation_features, training_labels, validation_labels = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(training_features)}, Validation size: {len(validation_features)}")

    pipeline = build_pipeline()
    pipeline.fit(training_features, training_labels)

    predicted_classes = pipeline.predict(validation_features)
    predicted_survival_probability = pipeline.predict_proba(validation_features)[:, 1]

    accuracy = accuracy_score(validation_labels, predicted_classes)
    correct_ranking_probability = roc_auc_score(validation_labels, predicted_survival_probability)

    print(f"Validation Accuracy:                    {accuracy:.4f}")
    print(f"Correct Ranking Probability (ROC AUC):  {correct_ranking_probability:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Pipeline saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
