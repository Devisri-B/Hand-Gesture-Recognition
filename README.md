# Hand-Gesture-Recognition
This project aims to recognize hand gestures using a deep learning model trained on a dataset of hand images. The project leverages data augmentation and trains a convolutional neural network (CNN) for gesture classification into three classes

## Project Overview
The goal of this project is to classify hand gestures into one of three categories:
Class 1: Touch
Class 2: No Hands
Class 3: No Touch with Hands

## Dataset Structure
The dataset is divided into three subsets: train, validation, and test. Each subset contains images organized by class labels.
- train: Used for model training.
- validation: Used for model validation during training.
- test: Used for final model evaluation.
Classes
- Class 1 Touch
- Class 2 No Hands
- Class 3 No Touch with Hands

## Preprocessing
The following data augmentation techniques are used during training:
- Rescaling pixel values to [0, 1]
- Random rotations, width/height shifts, and shear transformations
- Zooming, flipping (horizontal and vertical), and random brightness adjustments

## Model Architecture
The model architecture leverages a convolutional neural network (CNN) optimized for image classification tasks. Details of the architecture, including layers, activation functions, and hyperparameters, can be modified as required.

## Usage Requirements
Python 3.x, 
TensorFlow,
Keras,
NumPy,
OpenCV,
Jupyter Notebook (for running the provided notebook)

## Data Preparation
Ensure the dataset is organized in the following directory structure:
```vbnet
dataset/
    train/
        Class 1 Touch/
        Class 2 No Hands/
        Class 3 No Touch with Hands/
    validation/
        Class 1 Touch/
        Class 2 No Hands/
        Class 3 No Touch with Hands/
    test/
        Class 1 Touch/
        Class 2 No Hands/
        Class 3 No Touch with Hands/
```

## Training the Model
- Load the data using ImageDataGenerator for data augmentation.
- Train the model using the training data.
- Monitor validation accuracy and loss.
  
## Evaluating the Model
Once trained, evaluate the model using the test dataset to assess its performance.

## Results
During the training process, you can monitor the model's accuracy and loss. The dataset comprises:
Training Images: 38,503 images across three classes.
Validation Images: 8,252 images across three classes.
Test Images: 8,252 images across three classes.
Final results and accuracy will depend on hyperparameter tuning and model architecture.

