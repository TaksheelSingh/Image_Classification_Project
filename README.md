# Image Classification Project

## Overview
This project is an image classification model trained using a Convolutional Neural Network (CNN) on the CIFAR-10 dataset. It includes a Flask-based web application for image classification.

## Features
- Loads and preprocesses the CIFAR-10 dataset
- Trains a CNN model for image classification
- Saves the trained model for future predictions
- Implements a Flask web application for user-friendly interaction

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/TaksheelSingh/Image_Classification_Project.git
cd Image_Classification_Project
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
### 1. Train the Model
```bash
python train_model.py
```
This will train the CNN model and save it as `image_classifier_model.h5`.

### 2. Run the Web Application
```bash
python flask_app.py
```
The application will be available at `http://127.0.0.1:5000/`.

## Project Structure
```
Image_Classification_Project/
│── dataset_preprocessing.py  # Prepares dataset for training
│── train_model.py            # Trains the CNN model
│── cnn_model.py              # Defines the CNN architecture
│── image_classifier_model.h5 # Saved trained model
│── flask_app.py              # Web application for image classification
│── run_project.py            # Runs the entire pipeline
│── requirements.txt          # Required Python packages
│── .gitignore                # Files to exclude from version control
│── README.md                 # Project documentation
```

## Dependencies
- TensorFlow
- Keras
- Flask
- NumPy
- OpenCV
- Matplotlib

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Contributing
Feel free to contribute by creating pull requests. Suggestions and improvements are always welcome!

## License
This project is open-source and available under the MIT License.
