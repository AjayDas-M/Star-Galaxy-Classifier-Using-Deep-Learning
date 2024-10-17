import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Streamlit UI
st.title("Star-Galaxy Classification")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# If an image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert the image to a numpy array without additional preprocessing
    image_array = np.array(image)
    
    # Expand dimensions to match the input shape of the model (1, height, width, channels)
    processed_image = np.expand_dims(image_array, axis=0)
    
    # Load the model (adjust the path to your model file)
    model = load_model("/home/user/ajay das/mini-project/model/mymodel.h5")
    
    # Make the prediction
    try:
        prediction = model.predict(processed_image)
        # Assume the model outputs a probability for each class: [star_prob, galaxy_prob]
        class_names = ['Star', 'Galaxy']
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        st.success
