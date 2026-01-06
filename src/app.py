import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import DiabetesPreprocessor
import config

app = FastAPI(
    title="Diabetes Risk Prediction API",
    description="API untuk memprediksi risiko diabetes menggunakan XGBoost Model",
    version="1.0.0"
)

model = None
preprocessor = None

@app.on_event("startup")
def load_artifacts():
    global model, preprocessor
    
    processor_path = os.path.join(config.MODEL_DIR, "processor.pkl")
    if os.path.exists(processor_path):
        preprocessor = joblib.load(processor_path)
        print(f"[INFO] Preprocessor loaded from {processor_path}")
    else:
        print(f"[WARNING] Preprocessor not found at {processor_path}")

    model_path = os.path.join(config.MODEL_DIR, "model_xgb.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"[INFO] Model loaded from {model_path}")
    else:
        print(f"[WARNING] Model not found at {model_path}")

class PatientData(BaseModel):
    id: int
    age: int
    alcohol_consumption_per_week: int
    physical_activity_minutes_per_week: int
    diet_score: float
    sleep_hours_per_day: float
    screen_time_hours_per_day: float
    bmi: float
    waist_to_hip_ratio: float
    systolic_bp: int
    diastolic_bp: int
    heart_rate: int
    cholesterol_total: int
    hdl_cholesterol: int
    ldl_cholesterol: int
    triglycerides: int
    gender: str
    ethnicity: str
    education_level: str
    income_level: str
    smoking_status: str
    employment_status: str
    family_history_diabetes: int
    hypertension_history: int
    cardiovascular_history: int

@app.get("/")
def index():
    return {
        "message": "Diabetes Prediction API is Running!"
    }

@app.post("/predict")
def predict_diabetes(data: PatientData):
    global model, preprocessor
    
    if not model or not preprocessor:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded.")

    try:
        input_data = pd.DataFrame([data.model_dump(exclude={'id'})])
        processed_data = preprocessor.transform(input_data)
        
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        
        return {
            "status": "success",
            "prediction": int(prediction),
            "probability": float(round(probability, 4)),
            "risk_label": "High Risk" if probability > 0.5 else "Low Risk"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)