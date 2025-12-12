import os
import pickle

# Identify base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # /app/app
PROJECT_ROOT = os.path.dirname(BASE_DIR)                   # /app

# Model folder: /app/ml_models/models/
MODEL_DIR = os.path.join(PROJECT_ROOT, "ml_models", "models")

# Global dictionary for models
MODELS = {}

def load_single_model(path):
    """Loads one .pkl file."""
    with open(path, "rb") as f:
        return pickle.load(f)

def load_all_models():
    """Loads ALL models inside ml_models/models."""
    global MODELS
    MODELS = {}

    if not os.path.exists(MODEL_DIR):
        print("Model directory not found:", MODEL_DIR)
        return MODELS

    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(".pkl"):
            model_name = filename.replace(".pkl", "")
            model_path = os.path.join(MODEL_DIR, filename)

            try:
                MODELS[model_name] = load_single_model(model_path)
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

    print("Available models:", list(MODELS.keys()))
    return MODELS

# LOAD MODELS AT STARTUP
load_all_models()
