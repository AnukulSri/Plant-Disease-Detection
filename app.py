from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from keras.applications.resnet50 import preprocess_input
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
model = None
class_names = ["Apple Scab Leaf", "Apple leaf", "Apple rust leaf", "Bell_pepper leaf", "Bell_pepper leaf spot",
               "Blueberry leaf", "Cherry leaf", "Corn Gray leaf spot", "Corn leaf blight", "Corn rust leaf",
               "Peach leaf", "Potato leaf early blight", "Potato leaf late blight", "Raspberry leaf", "Soyabean leaf",
               "Squash Powdery mildew leaf", "Strawberry leaf", "Tomato Early blight leaf", "Tomato Septoria leaf spot",
               "Tomato leaf", "Tomato leaf bacterial spot", "Tomato leaf late blight", "Tomato leaf mosaic virus",
               "Tomato leaf yellow virus", "Tomato mold leaf", "Tomato two spotted spider mites leaf", "grape leaf",
               "grape leaf black rot"]

def load_model():
    global model
    model = tf.keras.models.load_model('plant_leaf_classifier')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            try:
                img = Image.open(file.stream).convert('RGB')
                img = img.resize((180, 180))
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                prediction = model.predict(img_array)
                predicted_class = class_names[np.argmax(prediction)]

                return jsonify({'prediction': predicted_class})

            except Exception as e:
                return jsonify({'error': str(e)})

if __name__ == '__main__':
    load_model()
    app.run(debug=True)
