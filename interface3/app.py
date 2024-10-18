import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the pre-trained model (update with your model path)
model = tf.keras.models.load_model('C:/Users/user/Documents/mini-project/model/star_galaxy_classification_model.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((128, 128))  # Resize according to your model's input size
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to classify the image
def classify_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    try:
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)
        class_label = "Galaxy" if prediction[0][0] > 0.5 else "Star"  # Adjust threshold based on your model
        result_label.config(text=f"Prediction: {class_label}")
        
        # Display the selected image
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Creating the main application window
root = tk.Tk()
root.title("Star-Galaxy Classification")

# Button to upload an image
upload_button = tk.Button(root, text="Upload Image", command=classify_image)
upload_button.pack(pady=10)

# Label to display the result
result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 14))
result_label.pack(pady=10)

# Label to display the selected image
image_label = tk.Label(root)
image_label.pack()

# Run the application
root.mainloop()
