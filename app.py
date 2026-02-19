from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI(title="Customer Segmentation API")

# Load artifacts
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
lda = pickle.load(open('lda.pkl', 'rb'))
kmeans_model = pickle.load(open('kmeans_model.pkl', 'rb'))
clf_model = pickle.load(open('classifier_model.pkl', 'rb'))  # RF or XGB

FEATURE_COLS = [
    'Age','Education','Marital Status','Parental Status','Children','Income',
    'Total_Spending','Days_as_Customer','Recency','Wines','Fruits','Meat','Fish',
    'Sweets','Gold','Web','Catalog','Store','Discount Purchases','Total Promo',
    'NumWebVisitsMonth','Family_Size','Spending_per_Day','Digital_Engagement',
    'Offline_Engagement','Discount_Ratio','Premium_Ratio','Freshness_Score','Variety_Index'
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

@app.post("/predict")
def predict_cluster(payload: CustomerInput):
    data = pd.DataFrame([payload.dict()])
    
    # align column names with training features
    data.columns = [
        'Age','Education','Marital Status','Parental Status','Children','Income',
        'Total_Spending','Days_as_Customer','Recency','Wines','Fruits','Meat','Fish',
        'Sweets','Gold','Web','Catalog','Store','Discount Purchases','Total Promo',
        'NumWebVisitsMonth','Family_Size','Spending_per_Day','Digital_Engagement',
        'Offline_Engagement','Discount_Ratio','Premium_Ratio','Freshness_Score','Variety_Index'
    ]
    
    X_scaled = preprocessor.transform(data)
    X_lda = lda.transform(X_scaled)
    
    cluster_unsupervised = int(kmeans_model.predict(X_lda)[0])
    cluster_supervised = int(clf_model.predict(X_scaled)[0])
    
    return {
        "kmeans_cluster": cluster_unsupervised,
        "predicted_cluster": cluster_supervised
    }

@app.get("/")
def home():
    return {"message": "Customer Segmentation API is running 🚀"}
