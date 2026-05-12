from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io



import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "cattle_model.h5")
model = tf.keras.models.load_model(model_path)


import cv2
import base64
from tensorflow.keras import models

def generate_gradcam(img_array, model, last_conv_layer_name="out_relu"):
    # 1. Create a model that maps the input image to the activations of the last conv layer
    grad_model = models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Get the gradients for the predicted class
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 3. This is the gradient of the output class with regard to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 4. Mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, original_img):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Resize to match original image
    jet = cv2.resize(jet, (original_img.shape[1], original_img.shape[0]))
    
    # Superimpose the heatmap on original image
    overlayed_img = jet * 0.4 + original_img
    _, buffer = cv2.imencode('.jpg', overlayed_img)
    return base64.b64encode(buffer).decode('utf-8')
app = FastAPI()

# IMPORTANT: Replace "*" with your Render frontend URL for security later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://the-wild-lens.netlify.app/"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (make sure cattle_model.h5 is in the same folder)
model = tf.keras.models.load_model('cattle_model.h5')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert('RGB')
    
    # Preprocess: MobileNetV2 expects 224x224
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Inference
    predictions = model.predict(img_array)
    classes = ['Buffalo', 'Cattle'] # Ensure this order matches your training
    
    predicted_class = classes[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100

    # --- ADD THESE TWO LINES TO ACTUALLY GENERATE THE HEATMAP ---
    heatmap_raw = generate_gradcam(img_array, model)
    heatmap_base64 = overlay_heatmap(heatmap_raw, np.array(img))
    # ------------------------------------------------------------

    return {
        "class": predicted_class,
        "confidence": round(confidence, 2),
        "heatmap": f"data:image/jpeg;base64,{heatmap_base64}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
