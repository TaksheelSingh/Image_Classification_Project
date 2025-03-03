from flask import Flask, request, render_template
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model("image_classifier_model.h5")

# Function to preprocess the uploaded image
def preprocess_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file was uploaded in the POST request
        if "file" not in request.files:
            return "No file uploaded", 400
        
        file = request.files["file"]

        # Check if the file is empty
        if file.filename == "":
            return "Empty file name", 400
        
        # Check if the file is an allowed image type
        if file:
            img = preprocess_image(file)

            # Get prediction from the model
            prediction = np.argmax(model.predict(img))
            return f"Predicted Class: {prediction}"

    # Render the index.html template from the 'templates' folder
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
