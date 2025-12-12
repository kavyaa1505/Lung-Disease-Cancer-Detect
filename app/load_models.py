import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app/
PROJECT_ROOT = os.path.dirname(BASE_DIR)               # project root

MODEL_DIR = os.path.join(PROJECT_ROOT, "ml_models", "models")

def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

# Load all models automatically
MODELS = {}

if os.path.exists(MODEL_DIR):
    for file in os.listdir(MODEL_DIR):
        if file.endswith(".pkl"):
            full_path = os.path.join(MODEL_DIR, file)
            model_name = file.replace(".pkl", "")
            try:
                MODELS[model_name] = load_model(full_path)
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Failed to load {file}: {e}")
else:
    print("Model directory not found:", MODEL_DIR)

print("Available models:", list(MODELS.keys()))
