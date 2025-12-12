from PIL import Image
import numpy as np
import io

async def preprocess_image(file):
    image_data = await file.read()
    img = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Resize to match your model input shape
    img = img.resize((224, 224))

    # Convert to numpy
    arr = np.array(img).flatten() / 255.0
    return arr

