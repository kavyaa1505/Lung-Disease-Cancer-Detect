from fastapi import FastAPI, File, UploadFile
from app.router import route_prediction
from app.load_models import load_all_models, MODELS

app = FastAPI()

# Load models at startup
load_all_models()

@app.get("/")
def home():
    return {"message": "ML API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return await route_prediction(file)
