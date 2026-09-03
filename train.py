from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from data_preprocessing import prepare_data

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "Bank_Churn.csv"
MODEL_PATH = PROJECT_ROOT / "smartretain_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "smartretain_preprocessor.joblib"


def build_preprocessor(features):
    numeric_features = features.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]

    return ColumnTransformer(
        transformers=[
            ("numbers", StandardScaler(), numeric_features),
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )


def train_models(X_train, X_test, y_train, y_test):
    X_train_balanced, y_train_balanced = SMOTE(random_state=42).fit_resample(
        X_train, y_train
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    model_results = {}
    for model_name, model in models.items():
        model.fit(X_train_balanced, y_train_balanced)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }
        model_results[model_name] = metrics

        print(f"\n=== {model_name} ===")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name.title()}: {metric_value:.4f}")
        print(classification_report(y_test, predictions, target_names=["No Churn", "Churn"]))

    return model_results


def tune_model(model_name, X_train, y_train):
    model_options = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=42),
            {"C": [0.1, 1, 10], "solver": ["liblinear", "lbfgs"]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=42),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            },
        ),
    }
    model, parameter_grid = model_options[model_name]
    X_train_balanced, y_train_balanced = SMOTE(random_state=42).fit_resample(
        X_train, y_train
    )

    grid_search = GridSearchCV(
        model,
        parameter_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
    )
    grid_search.fit(X_train_balanced, y_train_balanced)
    print(f"\nBest parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def main():
    X, y = prepare_data(DATASET_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    print(f"\nPrepared feature count: {X_train_scaled.shape[1]}")

    model_results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    best_model_name = max(model_results, key=lambda name: model_results[name]["roc_auc"])
    print(f"Selected model: {best_model_name}")

    best_rf_model = tune_model(best_model_name, X_train_scaled, y_train)
    predictions = best_rf_model.predict(X_test_scaled)
    probabilities = best_rf_model.predict_proba(X_test_scaled)[:, 1]
    print("\n=== Tuned Model Evaluation ===")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.4f}")
    print("Recall matters here because missing a customer who is likely to churn can mean losing the chance to retain them.")

    joblib.dump(best_rf_model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Preprocessor saved to: {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    main()
