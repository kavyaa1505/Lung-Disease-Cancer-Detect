import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app/
PROJECT_ROOT = os.path.dirname(BASE_DIR)               # root folder

MODEL_DIR = os.path.join(PROJECT_ROOT, "ml_models", "models")

def load_model(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        print("Model not found:", path)
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
