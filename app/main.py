from fastapi import FastAPI
from app.router import route_prediction
from app.load_models import load_all_models , MODELS

app = FastAPI()

# Load models at startup
load_all_models()

@app.get("/")
def home():
    return {"message": "ML API is running"}

@app.post("/predict")
async def predict(text: str):
    """
    Accept a plain text string and return model prediction.
    """
    return await route_prediction(text)
