import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)
model = load_model('star_galaxy_classifier.h5')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('upload.html')

# Define a route for the upload page
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    img_path = os.path.join('static', file.filename)
    file.save(img_path)
    
    # Preprocess the image
    img = Image.open(img_path).resize((224, 224))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make predictions
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)[0]
    
    # Map class index to class name
    class_labels = {0: 'Star', 1: 'Planet'}  # Adjust based on your model's output
    result = class_labels.get(class_index, 'Unknown')

    return render_template('result.html', result=result, image_file=file.filename)

if __name__ == '__main__':
    app.run(debug=True , port=5001)
