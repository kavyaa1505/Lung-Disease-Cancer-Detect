from fastapi import UploadFile
from app.load_models import MODELS
from app.preprocess import preprocess_image

async def route_prediction(file: UploadFile):
    filename = file.filename.lower()
    image = await preprocess_image(file)

    # Basic routing logic
    if "xray" in filename:
        model = MODELS.get("xray_model.pkl")
    elif "ct" in filename:
        model = MODELS.get("ct_model.pkl")
    else:
        return {"error": "Filename must include either 'xray' or 'ct'"}

    if model is None:
        return {"error": "Required model not found on server."}

    # For sklearn models
    try:
        prediction = model.predict([image])
        return {"prediction": prediction[0]}
    except:
        return {"error": "Prediction failed. Model may require PyTorch logic."}
