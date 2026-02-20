from flask import Flask, jsonify, request
import os
import joblib
import pandas as pd

# Try to import Cassandra session logic from your db.py
try:
    from db import get_session
except ImportError:
    get_session = None

app = Flask(__name__)

# Global variables for models and database
MODELS = {}
SESSION = None
MODELS_LOADED = False

# Feature columns (Must match the order used during model training)
FEATURE_COLS = [
    "Age", "Education", "Marital Status", "Parental Status", "Children", 
    "Income", "Total_Spending", "Days_as_Customer", "Recency", "Wines", 
    "Fruits", "Meat", "Fish", "Sweets", "Gold", "Web", "Catalog", 
    "Store", "Discount Purchases", "Total Promo", "NumWebVisitsMonth", 
    "Family_Size", "Spending_per_Day", "Digital_Engagement", 
    "Offline_Engagement", "Discount_Ratio", "Premium_Ratio", 
    "Freshness_Score", "Variety_Index"
]

def init_resources():
    """Initializes models and DB. Skips during CircleCI builds."""
    global MODELS, SESSION, MODELS_LOADED
    
    if os.getenv("CI"):
        print("CI mode: skipping heavy resource loading")
        return

    # Initialize Cassandra
    if get_session:
        try:
            SESSION = get_session()
            print("Cassandra connected successfully")
        except Exception as e:
            print(f"Cassandra connection failed: {e}")

    # Load Pickle Models
    try:
        MODELS["classifier"] = joblib.load("classifier_model.pkl")
        MODELS["kmeans"] = joblib.load("kmeans_model.pkl")
        MODELS["lda"] = joblib.load("lda.pkl")
        MODELS["preprocessor"] = joblib.load("preprocessor.pkl")
        MODELS_LOADED = True
        print("All models loaded successfully")
    except Exception as e:
        print(f"Model loading failed: {e}")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "Customer Segmentation API running (Flask)"})

@app.route("/predict", methods=["POST"])
def predict():
    if os.getenv("CI"):
        return jsonify({"message": "CI mode: prediction skipped"}), 200
    
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded on server"}), 503

    # Get JSON data from request
    input_data = request.get_json()
    
    try:
        # Convert to DataFrame and fix column names
        df = pd.DataFrame([input_data])
        # Ensure your input JSON keys match the FEATURE_COLS or rename them here
        
        # 1. Preprocess & Scale
        x_scaled = MODELS["preprocessor"].transform(df)
        
        # 2. Dimensionality Reduction (LDA)
        x_lda = MODELS["lda"].transform(x_scaled)

        # 3. Predict Clusters
        cluster_unsupervised = int(MODELS["kmeans"].predict(x_lda)[0])
        cluster_supervised = int(MODELS["classifier"].predict(x_scaled)[0])

        return jsonify({
            "kmeans_cluster": cluster_unsupervised,
            "predicted_cluster": cluster_supervised,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Load resources before starting the server
    init_resources()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)