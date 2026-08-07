import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = BASE_DIR / "sleep_disorder_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI()

GENDER_MAP = {"Male": 1, "Female": 0}
BMI_MAP = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}


class PredictionRequest(BaseModel):
    gender: str = Field(pattern="^(Male|Female)$")
    age: int = Field(ge=18, le=80)
    sleep_duration: float = Field(ge=4.0, le=10.0)
    quality_of_sleep: int = Field(ge=1, le=10)
    physical_activity: int = Field(ge=0, le=120)
    stress_level: int = Field(ge=1, le=10)
    bmi_category: str = Field(pattern="^(Underweight|Normal|Overweight|Obese)$")
    heart_rate: int = Field(ge=40, le=120)
    daily_steps: int = Field(ge=0, le=20000)
    bp_systolic: int = Field(ge=80, le=200)
    bp_diastolic: int = Field(ge=50, le=130)


def build_features(payload: PredictionRequest) -> pd.DataFrame:
    return pd.DataFrame([{
        "Gender": GENDER_MAP[payload.gender],
        "Age": payload.age,
        "Sleep Duration": payload.sleep_duration,
        "Quality of Sleep": payload.quality_of_sleep,
        "Physical Activity Level": payload.physical_activity,
        "Stress Level": payload.stress_level,
        "BMI Category": BMI_MAP[payload.bmi_category],
        "Heart Rate": payload.heart_rate,
        "Daily Steps": payload.daily_steps,
        "BP Systolic": payload.bp_systolic,
        "BP Diastolic": payload.bp_diastolic,
    }])


@app.post("/predict")
def predict(payload: PredictionRequest):
    features = build_features(payload)
    result = model.predict(features)[0]
    return {"result": result}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
