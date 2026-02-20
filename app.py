import os
import joblib

if not os.getenv("CI"):
    classifier = joblib.load("classifier_model.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    lda = joblib.load("lda.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
else:
    print("CI detected – skipping model loading")