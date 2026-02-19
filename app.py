import os
import pickle

if os.getenv("LOAD_MODELS", "true") == "true":
    preprocessor = pickle.load(open("preprocessor.pkl", "rb"))
    lda = pickle.load(open("lda.pkl", "rb"))
    kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
    classifier = pickle.load(open("classifier_model.pkl", "rb"))
