from flask import Flask, render_template, request
from PIL import Image
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model('star_galaxy_classifier.h5')

def preprocess_image(image):
    image = image.resize((64, 64))  # Resize to match the model input size
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    class_names = ['Star', 'Galaxy']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            image = Image.open(file.stream)
            predicted_class, confidence = predict(image)
            return render_template('index.html', prediction=f"{predicted_class} ({confidence:.2f}%)")
    
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
