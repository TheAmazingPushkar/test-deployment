from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os
import cv2
import base64
from tensorflow.keras import models

# --- SETUP PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure the folder name is 'model' (singular) as per your directory structure
model_path = os.path.join(BASE_DIR, "model", "cattle_model.h5")

# Check if model exists before trying to load (helps debug Render logs)
if not os.path.exists(model_path):
    print(f"ERROR: Model file not found at {model_path}")
    model = None
else:
    model = tf.keras.models.load_model(model_path)

import gc

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded on server"}

    try:
        # Read and resize BEFORE creating heavy numpy arrays
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert('RGB')
        img_resized = img.resize((224, 224))
        
        # Convert to float32 to save memory over float64
        img_array = np.array(img_resized).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 1. Prediction
        predictions = model.predict(img_array)
        classes = ['Buffalo', 'Cattle']
        pred_idx = np.argmax(predictions[0])
        
        predicted_class = classes[pred_idx]
        confidence = float(predictions[0][pred_idx]) * 100

        # 2. Grad-CAM (Wrapped in try/except so it doesn't kill the whole request)
        heatmap_base64 = ""
        try:
            heatmap_raw = generate_gradcam(img_array, model)
            heatmap_base64 = overlay_heatmap(heatmap_raw, np.array(img_resized))
        except Exception as grad_err:
            print(f"Grad-CAM failed: {grad_err}")
            # We continue even if heatmap fails so you at least get the label

        # 3. Memory Cleanup
        del img
        del img_array
        gc.collect()

        return {
            "class": predicted_class,
            "confidence": round(confidence, 2),
            "heatmap": f"data:image/jpeg;base64,{heatmap_base64}" if heatmap_base64 else None
        }

    except Exception as e:
        # This sends the ACTUAL error to your browser console
        return {"error": str(e), "traceback": "Check Render Logs"}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded on server"}

    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert('RGB')
    
    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Inference
    predictions = model.predict(img_array)
    classes = ['Buffalo', 'Cattle']
    
    predicted_class = classes[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100

    # Heatmap logic - using img_resized to ensure shapes match for overlay
    heatmap_raw = generate_gradcam(img_array, model)
    heatmap_base64 = overlay_heatmap(heatmap_raw, np.array(img_resized))

    return {
        "class": predicted_class,
        "confidence": round(confidence, 2),
        "heatmap": f"data:image/jpeg;base64,{heatmap_base64}"
    }

# Render uses the uvicorn command in the dashboard, so this block is mostly for local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
