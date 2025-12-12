from fastapi import HTTPException
from app.load_models import MODELS

async def route_prediction(text: str):
    try:
        model = MODELS.get("tokenizer")  # or your actual model name
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Predict using the ML model
        prediction = model.predict([text])[0]  
        return {"prediction": str(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
