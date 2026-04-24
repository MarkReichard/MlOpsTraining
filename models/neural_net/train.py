"""
train.py — Neural Network training script (Titanic survival)
=============================================================
Trains a Multi-Layer Perceptron (MLP) on the same Titanic data as the Random
Forest trainer and saves the pipeline in the identical format so the shared
prediction API can load either model interchangeably.

What is a Multi-Layer Perceptron?
  An MLP is the classic "neural network" — a chain of layers where each layer
  is a set of neurons that each compute a weighted sum of their inputs, apply a
  non-linear function (ReLU), and pass the result forward.  This implementation
  has two hidden layers: 64 neurons then 32 neurons.  Training adjusts the
  weights using backpropagation (gradient descent).

Key difference from the Random Forest approach:
  Neural networks are sensitive to the *scale* of input values.  If one column
  has values 0–512 (Fare) and another has 0–3 (Pclass), the large-valued column
  can dominate the gradient updates.  To fix this, we add a StandardScaler step
  that normalises each numeric column to mean=0 and standard deviation=1 before
  the network sees it.  The Random Forest does not need this because it only
  cares about the *order* of values, not their magnitude.

Output: /artifacts/pipeline.joblib
  Same format as the Random Forest pipeline — the API loads it without knowing
  or caring which model is inside.
"""

import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths — overridable via environment variables.
# ---------------------------------------------------------------------------
DATA_PATH = os.environ.get("DATA_PATH", "/data/train.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", "/artifacts/pipeline.joblib")

# ---------------------------------------------------------------------------
# Feature definitions (identical to the Random Forest trainer).
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["PassengerClass", "Age", "SiblingsOrSpouses", "ParentsOrChildren", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "PortOfEmbarkation"]
TARGET = "Survived"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Network architecture: two hidden layers with 64 and 32 neurons respectively.
# More neurons = more capacity to learn complex patterns, but slower to train
# and more likely to overfit on small datasets.
HIDDEN_LAYER_SIZES = (64, 32)

# Maximum number of training passes over the full dataset.
# If the loss stops improving before this, training stops early.
MAX_ITER = 500


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
    Construct the preprocessing + MLP classifier pipeline.

    Numeric preprocessing adds StandardScaler after median imputation.
    StandardScaler subtracts the mean and divides by the standard deviation of
    each column so all numeric inputs have roughly the same range.  This is
    essential for gradient-based optimisers like the one inside MLPClassifier.

    Categorical preprocessing is identical to the Random Forest: fill missing
    values with the most common category, then one-hot encode.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        # Normalise to mean=0, std=1.  Neural networks converge much faster
        # and more reliably when input features are on the same scale.
        ("scaler", StandardScaler()),
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
        ("classifier", MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYER_SIZES,
            max_iter=MAX_ITER,
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
