from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import json
import os

app = FastAPI()

model = joblib.load("models/best_model.pkl")

class InputData(BaseModel):
    soil_moisture_pct: float = Field(..., ge=10, le=60)
    crop_type_index: int = Field(..., ge=1, le=5)
    field_size_hectares: float = Field(..., ge=1, le=50)
    temperature_c: float = Field(..., ge=20, le=45)

@app.get("/health")
def health():
    return {"status": "operational", "service": "CropSense API"}

@app.post("/estimate")
def estimate(data: InputData):
    try:
        features = np.array([[ 
            data.soil_moisture_pct,
            data.crop_type_index,
            data.field_size_hectares,
            data.temperature_c
        ]])

        pred = float(model.predict(features)[0])

        # ✅ SAVE JSON (IMPORTANT)
        os.makedirs("results", exist_ok=True)

        output = {
            "health_endpoint": "/health",
            "predict_endpoint": "/estimate",
            "port": 8500,
            "health_response": {"status": "operational", "service": "CropSense API"},
            "test_input": data.dict(),
            "prediction": pred
        }

        with open("results/step3_s4.json", "w") as f:
            json.dump(output, f, indent=4)

        return {"prediction": pred}

    except Exception:
        raise HTTPException(status_code=422, detail="Invalid input")