import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_DIR = os.path.join(PROJECT_ROOT, "ml_models", "models")

MODELS = {}

def load_single_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_all_models():
    """Loads all .pkl models from ml_models/models."""
    global MODELS
    MODELS = {}

    if not os.path.exists(MODEL_DIR):
        print("Model directory not found:", MODEL_DIR)
        return MODELS

    for file in os.listdir(MODEL_DIR):
        if file.endswith(".pkl"):
            full_path = os.path.join(MODEL_DIR, file)
            model_name = file.replace(".pkl", "")

            try:
                model = load_single_model(full_path)
                MODELS[model_name] = model
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Failed to load {file}: {e}")

    print("Available models:", list(MODELS.keys()))
    return MODELS


# Load at startup
load_all_models()
