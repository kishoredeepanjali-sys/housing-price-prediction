import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

app = FastAPI(title="Housing Price Prediction API")

try:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
except FileNotFoundError:
    model = None
    pipeline = None


class HousingData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str


@app.get("/")
def read_root():
    return {"message": "Housing Price Prediction API is running."}


@app.get("/health")
def health_check():
    ready = model is not None and pipeline is not None
    return {"status": "ok" if ready else "model not loaded"}


@app.post("/predict")
def predict(data: HousingData):
    if model is None or pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="Model or pipeline not found. Make sure model.pkl and pipeline.pkl are present.",
        )

    input_df = pd.DataFrame([data.dict()])
    transformed = pipeline.transform(input_df)
    prediction = model.predict(transformed)

    return {"predicted_median_house_value": float(prediction[0])}


@app.post("/predict-batch")
def predict_batch(data: list[HousingData]):
    if model is None or pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="Model or pipeline not found. Make sure model.pkl and pipeline.pkl are present.",
        )

    input_df = pd.DataFrame([d.dict() for d in data])
    transformed = pipeline.transform(input_df)
    predictions = model.predict(transformed)

    return {"predictions": [float(p) for p in predictions]}