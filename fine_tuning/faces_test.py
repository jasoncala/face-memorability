# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:44:39 2021
@author: anonymous
"""

# Import necessary libraries
import numpy as np  # Numerical computations
from PIL import Image  # Image processing library
from cv2 import resize  # OpenCV library for resizing images
import tensorflow as tf  # TensorFlow for deep learning
from sklearn.model_selection import train_test_split  # Utility to split data into training and testing sets
import multiprocessing as mp  # Library for parallel processing
import pandas as pd  # Library for data manipulation and analysis
import os  # OS module for file and directory operations
import csv  # CSV module for handling CSV files
from keras_vggface.vggface import VGGFace  # Pretrained VGGFace model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential  # Model creation and management

# Path to the Excel file containing memorability scores
mem_score_xlsx = "faces/Memorability Scores/memorability-scores.xlsx"

# Path to the directory containing face images
face_images = "faces/10k US Adult Faces Database/Face Images/"

# Define a custom loss function for regression (Euclidean distance loss)
def euclidean_distance_loss(y_true, y_pred):
    """
    Computes the Euclidean distance between true and predicted values.
    Used as the loss function for the model.
    """
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_pred - y_true), axis=-1))

# Function to load and preprocess a single image
def load_image(image_file):
    """
    Opens an image file, resizes it to 224x224 pixels, converts it to RGB, and returns it as a NumPy array.
    """
    image = Image.open(image_file).resize((224, 224)).convert("RGB")
    return np.array(image, dtype=np.uint8)

# Function to split the dataset into training, validation, and test sets
def load_split(split_file):
    """
    Reads the Excel file, splits the dataset into training and testing sets, and further processes them.
    """
    faces_ds = pd.read_excel(split_file)  # Load the dataset from an Excel file
    X_train, X_test = train_test_split(faces_ds, test_size=0.2, random_state=42)  # Split data into train and test sets
    scores_train = list(X_train['Hit Rate (HR)'])  # Memorability scores for training
    names_train = list(X_train['Filename'])  # Image filenames for training
    scores_test = list(X_test['Hit Rate (HR)'])  # Memorability scores for testing
    names_test = list(X_test['Filename'])  # Image filenames for testing
    
    # Adjusted False Alarm rates (not used here but prepared for future computations)
    FA_test = list(X_test['False Alarm Rate (FAR)'])
    FA_train = list(X_train['False Alarm Rate (FAR)'])
    
    # Create structured lists of [filename, score]
    X_train = [[names_train[i], scores_train[i]] for i in range(len(scores_train))]
    X_test = [[names_test[i], scores_test[i]] for i in range(len(scores_test))]
    
    # Further split data for training, validation, and testing
    X_train = X_train[:int(len(X_train) / 2)]
    X_valid = X_test[:int(len(X_test) / 2)]
    X_test = X_test[int(len(X_test) / 2):]
    
    return X_train

# Generator to yield batches of data for training or evaluation
def lamem_generator(split_file, batch_size=32):
    """
    Generates batches of image data and labels for training/testing the model.
    """
    num_samples = len(split_file)
    for offset in range(0, num_samples, batch_size):
        if offset + batch_size > num_samples:
            batch_samples = split_file[offset:]  # Handle last batch with fewer samples
        else:
            batch_samples = split_file[offset:offset + batch_size]
        
        # Load images in parallel using multiprocessing
        inputs = mp.Pool().map(load_image, [face_images + i[0] for i in batch_samples])
        final_labels = [[i[1]] for i in batch_samples]  # Extract labels
        # Yield the batch of inputs (images) and labels
        yield ([np.array(inputs), np.array(inputs), np.array(inputs)], np.array(final_labels))

# List to store Spearman correlation coefficients for each evaluation
cors = []

# Loop over multiple saved model checkpoints for evaluation
for i in range(30):
    # Load the pretrained model for the specific epoch
    model = tf.keras.models.load_model("models/epoch_" + str(i + 5) + ".h5", custom_objects={'euclidean_distance_loss': euclidean_distance_loss})
    
    # Load and process the test split
    test_split = load_split(mem_score_xlsx)
    test_split = lamem_generator(test_split)  # Convert to data generator
    
    true_values = []  # To store true memorability scores
    predictions = []  # To store predicted scores
    
    # Evaluate the model on the test set
    for idx, (img, target) in enumerate(test_split):
        predict = model.predict(img)  # Predict scores for the batch of images
        true_values.append(target)  # Append true scores
        predictions.append(predict)  # Append predicted scores
        print(idx)
        print("True memorability score: " + str(target))
        print("Predicted memorability score: " + str(predict))
    
    # Convert predictions and true values to NumPy arrays
    y_pred = np.asarray(predictions[0])
    y_test = np.asarray(true_values[0])

    # Stack batches to create full predictions and true labels
    for i in range(len(predictions) - 1):
        y_pred = np.vstack((y_pred, np.asarray(predictions[i + 1])))
        y_test = np.vstack((y_test, np.asarray(true_values[i + 1])))

    # Calculate Spearman's correlation coefficient
    from scipy.stats import spearmanr
    coef, p = spearmanr(y_test, y_pred)
    
    # Compute mean squared error (MSE)
    temp = 0
    for i in range(len(y_pred)):
        temp += abs(y_pred[i] - y_test[i]) ** 2
    temp = temp / len(y_pred)
    
    print("error: " + str(temp))  # Print error
    print('Spearman’s correlation coefficient: %.3f' % coef)  # Print correlation
    cors.append(coef)  # Append correlation score

# Print average Spearman correlation coefficient across all evaluated models
print(np.mean(cors))
