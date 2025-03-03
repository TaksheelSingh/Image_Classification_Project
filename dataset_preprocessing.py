
import tensorflow as tf
from tensorflow.keras import datasets

# Load CIFAR-10 Dataset
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Normalize the images
X_train, X_test = X_train / 255.0, X_test / 255.0
print("Dataset Loaded and Normalized")
