import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model once
model = tf.keras.models.load_model('app/ml/model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # match training size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        return "Buffalo"
    else:
        return "Cow"
