import pandas as pd
import numpy as np
import json
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("irrigation_hours", axis=1)
y = df["irrigation_hours"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("cropsense-irrigation-hours")

results = []

models = {
    "SVR": SVR(),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

best_model = None
best_mae = float("inf")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_params(model.get_params())
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
        mlflow.set_tag("domain", "precision_farming")

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        })

        if mae < best_mae:
            best_mae = mae
            best_model = name
            joblib.dump(model, f"models/{name}.pkl")

output = {
    "experiment_name": "cropsense-irrigation-hours",
    "models": results,
    "best_model": best_model,
    "best_metric_name": "mae",
    "best_metric_value": best_mae
}

os.makedirs("results", exist_ok=True)
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 completed!")