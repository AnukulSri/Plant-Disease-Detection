import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('plant_leaf_classifier')
    return model

model = load_model()
class_names = ["Apple Scab Leaf", "Apple leaf", "Apple rust leaf", "Bell_pepper leaf", "Bell_pepper leaf spot",
               "Blueberry leaf", "Cherry leaf", "Corn Gray leaf spot", "Corn leaf blight", "Corn rust leaf",
               "Peach leaf", "Potato leaf early blight", "Potato leaf late blight", "Raspberry leaf", "Soyabean leaf",
               "Squash Powdery mildew leaf", "Strawberry leaf", "Tomato Early blight leaf", "Tomato Septoria leaf spot",
               "Tomato leaf", "Tomato leaf bacterial spot", "Tomato leaf late blight", "Tomato leaf mosaic virus",
               "Tomato leaf yellow virus", "Tomato mold leaf", "Tomato two spotted spider mites leaf", "grape leaf",
               "grape leaf black rot"]

# Streamlit app
st.title("Plant Leaf Classifier")
st.write("Upload an image of a plant leaf to classify its disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and preprocess the image
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize((180, 180))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        # Display the image and the prediction
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("Prediction:", predicted_class)

    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == '__main__':
    st._is_running_with_streamlit = True
    st.write('This app is running on Streamlit')
