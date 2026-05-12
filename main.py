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

def generate_gradcam(img_array, model, last_conv_layer_name="out_relu"):
    grad_model = models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10) # Added epsilon to prevent div by zero
    return heatmap.numpy()

def overlay_heatmap(heatmap, original_img):
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.resize(jet, (original_img.shape[1], original_img.shape[0]))
    overlayed_img = jet * 0.4 + original_img
    _, buffer = cv2.imencode('.jpg', overlayed_img)
    return base64.b64encode(buffer).decode('utf-8')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://the-wild-lens.netlify.app"],
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
