import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import random

# Load your pre-trained model
model = tf.keras.models.load_model('C:/Users/user/Documents/mini-project/model/star_galaxy_classification_model.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((64, 64))  # Resize to the input size of your model
    image = np.array(image) / 255.0  # Normalize the image
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Define a function for augmenting the image
def augment_image(image):
    # Random rotation
    if random.random() < 0.5:
        image = image.rotate(random.randint(-20, 20))
    
    # Random zoom
    if random.random() < 0.5:
        zoom_factor = random.uniform(0.85, 1.15)
        image = image.resize((int(64 * zoom_factor), int(64 * zoom_factor)))

    # Random horizontal flip
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Convert back to (64, 64) after augmentation if necessary
    return image.resize((64, 64))

# Title of the app
st.title("Star-Galaxy Classification")

# File uploader
uploaded_file = st.file_uploader("Upload an image of a star or galaxy", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Optionally augment the image
    augmented_image = augment_image(image)

    # Preprocess the image
    processed_image = preprocess_image(augmented_image)

    # Classify the image
    if st.button("Classify"):
        predictions = model.predict(processed_image)
        class_index = np.argmax(predictions, axis=1)
        
        # Display the result (modify according to your classes)
        if class_index == 0:
            st.write("Prediction: Star")
        else:
            st.write("Prediction: Galaxy")

# Add additional features or visualizations as needed
