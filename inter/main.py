import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('/home/user/ajay das/mini-project/model/my_model1.h5')
    return model

model = load_model()

# Title and description
st.title("Star-Galaxy Classification")
st.write("This application classifies images into either 'Star' or 'Galaxy' using a deep learning model.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image to match the model input requirements
    def preprocess_image(image):
        image = image.resize((64, 64))  # Adjust size based on your model's input
        image = np.array(image) / 255.0  # Normalize if necessary
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    processed_image = preprocess_image(image)

    # Make a prediction
    if st.button("Classify"):
        prediction = model.predict(processed_image)
        class_names = ['Star', 'Galaxy']
        predicted_class = class_names[np.argmax(prediction)]

        st.write(f"Prediction: **{predicted_class}**")
        st.write(f"Confidence: **{np.max(prediction) * 100:.2f}%**")
