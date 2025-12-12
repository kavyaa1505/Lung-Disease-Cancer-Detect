import os
import pickle
import torch

MODELS = {}

def load_all_models():
    model_dir = "ml_models/models"

    if not os.path.exists(model_dir):
        print("No models found in ml_models/models")
        return

    for file in os.listdir(model_dir):
        path = os.path.join(model_dir, file)

        # Load scikit-learn / pickle models
        if file.endswith(".pkl"):
            with open(path, "rb") as f:
                MODELS[file] = pickle.load(f)

        # Load PyTorch models
        elif file.endswith(".pt"):
            MODELS[file] = torch.load(path, map_location="cpu")

    print("Loaded models:", list(MODELS.keys()))
