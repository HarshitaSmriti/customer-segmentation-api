import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Customer Segmentation API")

MODELS_LOADED = False

if not os.getenv("CI"):
    classifier = joblib.load("classifier_model.pkl")
    kmeans_model = joblib.load("kmeans_model.pkl")
    lda = joblib.load("lda.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    MODELS_LOADED = True
else:
    print("CI detected - skipping model loading")

FEATURE_COLS = [
    "Age",
    "Education",
    "Marital Status",
    "Parental Status",
    "Children",
    "Income",
    "Total_Spending",
    "Days_as_Customer",
    "Recency",
    "Wines",
    "Fruits",
    "Meat",
    "Fish",
    "Sweets",
    "Gold",
    "Web",
    "Catalog",
    "Store",
    "Discount Purchases",
    "Total Promo",
    "NumWebVisitsMonth",
    "Family_Size",
    "Spending_per_Day",
    "Digital_Engagement",
    "Offline_Engagement",
    "Discount_Ratio",
    "Premium_Ratio",
    "Freshness_Score",
    "Variety_Index",
]


class CustomerInput(BaseModel):
    Age: int
    Education: int
    Marital_Status: int
    Parental_Status: int
    Children: int
    Income: float
    Total_Spending: float
    Days_as_Customer: int
    Recency: int
    Wines: float
    Fruits: float
    Meat: float
    Fish: float
    Sweets: float
    Gold: float
    Web: int
    Catalog: int
    Store: int
    Discount_Purchases: int
    Total_Promo: int
    NumWebVisitsMonth: int
    Family_Size: int
    Spending_per_Day: float
    Digital_Engagement: int
    Offline_Engagement: int
    Discount_Ratio: float
    Premium_Ratio: float
    Freshness_Score: float
    Variety_Index: int


@app.get("/")
def home():
    return {"message": "Customer Segmentation API is running"}


@app.post("/predict")
def predict_cluster(payload: CustomerInput):
    if not MODELS_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded. Set CI=false and restart service.",
        )

    data = pd.DataFrame([payload.model_dump()])
    data.columns = FEATURE_COLS

    x_scaled = preprocessor.transform(data)
    x_lda = lda.transform(x_scaled)

    cluster_unsupervised = int(kmeans_model.predict(x_lda)[0])
    cluster_supervised = int(classifier.predict(x_scaled)[0])

    return {
        "kmeans_cluster": cluster_unsupervised,
        "predicted_cluster": cluster_supervised,
    }
