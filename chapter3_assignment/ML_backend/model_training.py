import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "diamonds (1).csv"

class ModelTrainer:
    """Train and evaluate a Random Forest diamond-price regressor."""
    
    def __init__(self, n_estimators=100, max_depth=10, n_jobs=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
    def fit_and_evaluate(self, x_train, x_test, y_train, y_test):
        numeric_features = x_train.select_dtypes(include="number").columns.tolist()
        categorical_features = x_train.select_dtypes(exclude="number").columns.tolist()
        transformers = []
        if numeric_features:
            transformers.append(("numeric", "passthrough", numeric_features))
        if categorical_features:
            transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features))

        pipeline = Pipeline([
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            ("model", RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                n_jobs=self.n_jobs,
                random_state=42,
            )),
        ])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
        }
        return pipeline, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and track a diamond-price regression model with MLflow.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Diamond dataset CSV")
    parser.add_argument("--experiment", default="Diamond_Price_Prediction", help="MLflow experiment name")
    parser.add_argument("--run-name", default="RandomForest_Pipeline", help="MLflow run name")
    parser.add_argument("--tracking-uri", help="MLflow tracking URI; defaults to chapter3_assignment/mlflow.db")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees in the Random Forest")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum tree depth")
    parser.add_argument("--n-jobs", type=int, default=1, help="Training workers; use -1 to use all CPUs")
    args = parser.parse_args(argv)

    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        parser.error(f"CSV file does not exist: {data_path}")

    data = pd.read_csv(data_path)
    data = data.loc[:, ~data.columns.astype(str).str.startswith("Unnamed:")]
    if "price" not in data.columns:
        parser.error("CSV must contain a 'price' target column.")
    if data.empty or data.drop(columns="price").empty:
        parser.error("CSV must contain rows and at least one feature column.")
    if data.isna().any().any():
        parser.error("CSV must not contain missing values.")

    x = data.drop(columns="price")
    y = data["price"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    trainer = ModelTrainer(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=args.n_jobs,
    )

    chapter_dir = Path(__file__).resolve().parents[1]
    tracking_uri = args.tracking_uri or f"sqlite:///{(chapter_dir / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params({
            "dataset": data_path.name,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "n_jobs": args.n_jobs,
        })
        pipeline, test_metrics = trainer.fit_and_evaluate(
            x_train, x_test, y_train, y_test
        )
        mlflow.log_metrics(test_metrics)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="best_model",
            input_example=x_train.iloc[[0]],
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        mlflow.log_artifact(str(data_path), artifact_path="dataset")

        print(f"MLflow run logged: {run.info.run_id}")
        print(f"Experiment: {args.experiment}")
        print(f"Tracking URI: {tracking_uri}")
        print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()