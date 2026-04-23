"""
train.py — Titanic survival model training script.

Loads the Titanic CSV, preprocesses features, trains a RandomForestClassifier,
prints validation metrics, and saves the full preprocessing + model pipeline to
disk so the prediction API can load it without repeating any of this logic.

For a plain-English explanation of how RandomForestClassifier works and what it
is good for, see docs/random-forest.md.
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
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/pipeline.joblib")

# ---------------------------------------------------------------------------
# Feature definitions — kept here so train.py and predict_api.py both refer
# to the same column names without duplication.
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]
TARGET = "Survived"

# ---------------------------------------------------------------------------
# Hyperparameters — named constants instead of magic numbers so any change is
# visible in one place and self-documenting.
# ---------------------------------------------------------------------------
RANDOM_STATE = 42   # Controls the random choices made during training. Using a fixed
                    # number means you get the same model every time you run the script.
                    # Change it to any integer to get a different (equally valid) model.
TEST_SIZE = 0.2     # 20 % of rows held back for validation; 80 % used to train.
N_ESTIMATORS = 100  # Number of trees in the forest. More = more stable, slower.
MAX_DEPTH = 5       # Max levels per tree. Shallow trees generalise better on
                    # small datasets by avoiding memorising the training data.


def load_data(path: str) -> pd.DataFrame:
    """Load CSV and verify all required columns are present."""
    passenger_data = pd.read_csv(path)
    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [col for col in required_columns if col not in passenger_data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return passenger_data


def build_pipeline() -> Pipeline:
    """
    Construct the preprocessing + classifier pipeline.

    Why a Pipeline?
    Bundling preprocessing and the classifier into one object means:
    - fit() on training data and predict() on new data use exactly the same
      transformation logic — no risk of accidentally applying different steps.
    - The entire thing serialises to a single file with joblib, so the API
      only needs to load one artifact.

    Preprocessing steps:
    - Numeric columns: fill missing values with the column median.
      (Age and Fare occasionally have gaps in the Titanic dataset.)
    - Categorical columns: fill missing values with the most common value,
      then one-hot encode (convert e.g. "male"/"female" to 0/1 columns).
      handle_unknown="ignore" means an unseen category at inference time
      produces all-zero columns rather than raising an error.
    """
    # Fill gaps in numeric columns with the median of each column.
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Fill gaps in text columns, then convert text values to numbers.
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Apply the two transformers above to their respective column groups in
    # parallel, then concatenate the results into one feature matrix.
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    # Chain preprocessor → classifier into a single reusable object.
    # See docs/random-forest.md for an explanation of RandomForestClassifier.
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

    # Separate inputs from the label we want to predict.
    features = passenger_data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    labels = passenger_data[TARGET]

    # Split into training and validation sets.
    # The validation set is never seen during training; it gives an honest
    # estimate of how the model performs on new, unseen passengers.
    training_features, validation_features, training_labels, validation_labels = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(training_features)}, Validation size: {len(validation_features)}")

    pipeline = build_pipeline()
    pipeline.fit(training_features, training_labels)  # All learning happens here.

    predicted_classes = pipeline.predict(validation_features)           # Predicted outcome for each passenger: 1 = survived, 0 = did not survive.
    predicted_survival_probability = pipeline.predict_proba(validation_features)[:, 1]  # Model's confidence (0–1) that each passenger survived.

    # accuracy: percentage of validation passengers the model predicted correctly.
    # correct_ranking_probability: if you pick one survivor and one non-survivor at random,
    #   this is the probability the model gave the survivor a higher survival score.
    #   1.0 = always ranks survivors higher (perfect), 0.5 = no better than a coin flip.
    accuracy = accuracy_score(validation_labels, predicted_classes)
    # roc_auc_score compares the model's confidence scores against the true outcomes.
    # It asks: for every possible pair of (survivor, non-survivor), did the model score
    # the survivor higher? The result is the fraction of pairs where it got that right.
    correct_ranking_probability = roc_auc_score(validation_labels, predicted_survival_probability)

    print(f"Validation Accuracy:                    {accuracy:.4f}")
    print(f"Correct Ranking Probability (ROC AUC):  {correct_ranking_probability:.4f}")

    # Save the complete pipeline (preprocessing + model) to a single file.
    # The API container loads this file at startup and uses it for predictions.
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Pipeline saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
