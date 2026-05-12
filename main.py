from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os
import cv2
import base64
import gc

app = FastAPI()

# --- CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL LOADING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "cattle_model.h5")

if not os.path.exists(model_path):
    print(f"ERROR: Model file not found at {model_path}")
    model = None
else:
    model = tf.keras.models.load_model(model_path)

# --- HEATMAP UTILITIES ---

def generate_gradcam(img_array, model):
    """Generates a Grad-CAM heatmap for MobileNetV2 with a Sequential wrapper."""
    try:
        # Reach into the Sequential wrapper to get the MobileNetV2 base
        base_model = model.layers[0]
        
        # Target the last convolutional layer
        last_conv_layer = base_model.get_layer("out_relu")

        grad_model = tf.keras.models.Model(
            [base_model.inputs], [last_conv_layer.output, base_model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            class_channel = preds[:, 0]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    except Exception as e:
        print(f"Grad-CAM generation error: {e}")
        return None

def overlay_heatmap(heatmap, img_original):
    """Overlays the heatmap onto the image and returns base64 string."""
    if heatmap is None:
        return ""
    try:
        heatmap = cv2.resize(heatmap, (img_original.shape[1], img_original.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_original, 0.6, heatmap, 0.4, 0)
        _, buffer = cv2.imencode('.jpg', superimposed_img)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Overlay failed: {e}")
        return ""

# --- PREDICTION ENDPOINT ---

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded on server"}

    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert('RGB')
        
        # Match your training size: 244x244
        img_resized = img.resize((244, 244)) 
        img_array = np.array(img_resized).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Binary Prediction Logic
        prediction = model.predict(img_array)[0][0] 
        
        if prediction < 0.5:
            predicted_class = "It's a Buffalo"
            confidence = (1 - prediction) * 100
        else:
            predicted_class = "It's a cow"
            confidence = prediction * 100

        # Grad-CAM logic
        heatmap_base64 = ""
        try:
            heatmap_raw = generate_gradcam(img_array, model)
            if heatmap_raw is not None:
                heatmap_base64 = overlay_heatmap(heatmap_raw, np.array(img_resized))
        except Exception as e:
            print(f"Heatmap logic failed: {e}")

        # Clean up memory
        del img
        del img_array
        gc.collect()

        return {
            "class": predicted_class,
            "confidence": round(float(confidence), 2),
            "heatmap": f"data:image/jpeg;base64,{heatmap_base64}" if heatmap_base64 else None
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
