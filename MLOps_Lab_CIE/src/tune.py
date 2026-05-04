import pandas as pd
import numpy as np
import json
import mlflow
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

df = pd.read_csv("data/training_data.csv")

X = df.drop("irrigation_hours", axis=1)
y = df["irrigation_hours"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5]
}

mlflow.set_experiment("cropsense-irrigation-hours")

with mlflow.start_run(run_name="tuning-cropsense"):
    model = GradientBoostingRegressor(random_state=42)

    grid = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_absolute_error")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)
    best_mae = mean_absolute_error(y_test, preds)

    joblib.dump(best_model, "models/best_model.pkl")

    output = {
        "search_type": "grid",
        "n_folds": 5,
        "total_trials": len(grid.cv_results_["params"]),
        "best_params": grid.best_params_,
        "best_mae": best_mae,
        "best_cv_mae": -grid.best_score_,
        "parent_run_name": "tuning-cropsense"
    }

    with open("results/step2_s2.json", "w") as f:
        json.dump(output, f, indent=4)

print("Task 2 completed!")