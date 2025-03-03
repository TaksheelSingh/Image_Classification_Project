
import tensorflow as tf
from dataset_preprocessing import X_train, y_train
from cnn_model import create_model

model = create_model()
model.summary()

model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=32)
model.save("image_classifier_model.h5")
print("Model Trained and Saved")
