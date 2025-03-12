import os
import tensorflow as tf
from flask_app import app

if __name__ == "__main__":
    # Load the TensorFlow/Keras model
    model_path = "path/to/image_classifier_model.h5"  # Update with your actual model path
    model = tf.keras.models.load_model(model_path)

    # Set the model in the Flask app context
    app.config["MODEL"] = model

    # Run the Flask application
    app.run(debug=True)
