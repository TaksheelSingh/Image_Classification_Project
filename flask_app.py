
from flask import Flask, request, render_template
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model("image_classifier_model.h5")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img = request.files["file"]
        img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (32, 32))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = np.argmax(model.predict(img))
        return f"Predicted Class: {prediction}"

    return '''
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <button type="submit">Predict</button>
    </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)
