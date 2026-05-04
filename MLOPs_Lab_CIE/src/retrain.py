import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Load data
train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")

combined_df = pd.concat([train_df, new_df], ignore_index=True)

X = train_df.drop("irrigation_hours", axis=1)
y = train_df["irrigation_hours"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Champion model
champion = joblib.load("models/best_model.pkl")
champ_preds = champion.predict(X_test)
champ_rmse = np.sqrt(mean_squared_error(y_test, champ_preds))

# Retrain
X_new = combined_df.drop("irrigation_hours", axis=1)
y_new = combined_df["irrigation_hours"]

new_model = GradientBoostingRegressor(random_state=42)
new_model.fit(X_new, y_new)

new_preds = new_model.predict(X_test)
new_rmse = np.sqrt(mean_squared_error(y_test, new_preds))

improvement = champ_rmse - new_rmse

action = "promoted" if improvement > 0 else "kept_champion"

if action == "promoted":
    joblib.dump(new_model, "models/best_model.pkl")

output = {
    "original_data_rows": len(train_df),
    "new_data_rows": len(new_df),
    "combined_data_rows": len(combined_df),
    "champion_rmse": champ_rmse,
    "retrained_rmse": new_rmse,
    "improvement": improvement,
    "min_improvement_threshold": 0,
    "action": action,
    "comparison_metric": "rmse"
}

with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 completed!")