from flask import Flask, jsonify, request
import os
<<<<<<< HEAD

# Optional imports (don’t crash CI if missing heavy deps)
try:
    import joblib
except Exception:
    joblib = None

# Optional Cassandra hook (don’t crash CI)
try:
    from db import get_session
except Exception:
    get_session = None

app = Flask(__name__)

# Lazy-loaded globals
MODELS = {}
SESSION = None


def init_resources():
    """
    Initialize DB + models only when needed.
    Safe for CI (skips heavy stuff when CI=true).
    """
    global MODELS, SESSION

    if os.getenv("CI"):
        print("CI mode: skipping DB + model loading")
        return

    # Init Cassandra session (optional)
    if get_session:
        try:
            SESSION = get_session()
            print("Cassandra connected")
        except Exception as e:
            print("Cassandra connection failed:", e)

    # Load ML models (if available)
    if joblib:
        try:
            MODELS["classifier"] = joblib.load("classifier_model.pkl")
            MODELS["kmeans"] = joblib.load("kmeans_model.pkl")
            MODELS["lda"] = joblib.load("lda.pkl")          # LDA instead of PCA
            MODELS["preprocessor"] = joblib.load("preprocessor.pkl")
            print("Models loaded")
        except Exception as e:
            print("Model loading failed:", e)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "Customer Segmentation API running"})


@app.route("/train", methods=["GET"])
def train():
    """
    Trigger training pipeline.
    Hook this to your src/components pipeline.
    """
    init_resources()

    if os.getenv("CI"):
        return jsonify({"message": "CI mode: train endpoint skipped"}), 200

    # TODO: integrate with your pipeline
    # from src.components.model_trainer import ModelTrainer
    # ModelTrainer().train()

    return jsonify({"message": "Training triggered"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict customer cluster/personality.
    """
    init_resources()

    if os.getenv("CI"):
        return jsonify({"message": "CI mode: predict endpoint skipped"}), 200

    if not MODELS:
        return jsonify({"error": "Models not loaded"}), 500

    data = request.json

    # TODO: preprocess input using your pipeline
    # X = MODELS["preprocessor"].transform([data])

    # TODO: predict cluster
    # cluster = MODELS["kmeans"].predict(X)[0]
    # pred = MODELS["classifier"].predict(X)[0]

    return jsonify({
        "message": "Prediction endpoint wired",
        "note": "Connect this to your ML pipeline logic",
    }), 200


if __name__ == "__main__":
    # Don’t auto-run DB or models at import time
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
=======

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
>>>>>>> 365719a8fb816c1044589d3f06628dfd83afaf1b
