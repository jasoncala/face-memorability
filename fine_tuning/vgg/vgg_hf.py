# Import necessary libraries
import numpy as np
import pandas as pd
import PIL
from PIL import Image  # Image processing
from datetime import datetime, timedelta
import os
from tqdm import tqdm  # Progress bar
import csv
from sklearn.model_selection import train_test_split  # Splitting datasets
import random
import multiprocessing as mp  # Parallel processing
from keras_vggface.vggface import VGGFace  # Pretrained VGGFace model

import tensorflow as tf
from tensorflow.keras.layers import Dense  # Neural network layers
from tensorflow.keras.models import Model  # Model handling
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau  # Training callbacks

"""
This script fine-tunes a VGGFace model without dropout layers for memorability prediction.
It uses a dense output layer for regression and implements transfer learning. The script 
focuses on a simpler variation of the training process compared to `vgg_hf_drop.py`.
"""


# Define paths for memorability scores and face images
mem_score_xlsx = "faces/Memorability Scores/memorability-scores.xlsx"
face_images = "faces/10k US Adult Faces Database/Face Images/"

# Custom loss function (Euclidean distance loss) for regression tasks
def euclidean_distance_loss(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_pred - y_true), axis=-1))

# Load VGGFace model
model = VGGFace(model='vgg16')  # Use VGG16 architecture
outputs = model.layers[-2].output  # Access the second-to-last layer
output = Dense(1)(outputs)  # Add dense layer with one neuron for memorability score prediction
model = Model(inputs=model.inputs, outputs=output)  # Create modified model
print(model.summary())  # Print model architecture

# Function to freeze and unfreeze layers for fine-tuning
def setup_to_finetune(model, N_freeze_layer):
    """
    Freeze the bottom N_freeze_layer layers and keep the remaining layers trainable.
    This allows transfer learning by using the pretrained features from earlier layers.
    """
    for layer in model.layers[:N_freeze_layer]:
        layer.trainable = False  # Freeze these layers
    for layer in model.layers[N_freeze_layer:]:
        layer.trainable = True  # Unfreeze these layers for training

# Function to split the dataset into training, validation, and testing subsets
def load_split(split_file):
    faces_ds = pd.read_excel(split_file)  # Load dataset from Excel file
    X_train, X_test = train_test_split(faces_ds, test_size=0.2, random_state=42)  # 80-20 split
    # Process the scores by subtracting false alarm rates
    scores_train = list(X_train['Hit Rate (HR)'] - X_train['False Alarm Rate (FAR)'])
    names_train = list(X_train['Filename'])
    scores_test = list(X_test['Hit Rate (HR)'] - X_test['False Alarm Rate (FAR)'])
    names_test = list(X_test['Filename'])

    # Create structured train/test datasets as lists of [filename, adjusted_score]
    X_train = [[names_train[i], scores_train[i]] for i in range(len(scores_train))]
    X_test = [[names_test[i], scores_test[i]] for i in range(len(scores_test))]
    X_valid = X_test[:int(len(X_test)/2)]  # Split test set into validation and test subsets
    X_test = X_test[int(len(X_test)/2):]
    return X_train, X_test, X_valid

# Function to preprocess a single image
def load_image(image_file):
    image = Image.open(image_file).resize((224,224)).convert("RGB")  # Resize and convert to RGB
    return np.array(image, dtype=np.uint8)  # Return as NumPy array

# Generator function to create batches of images and labels
def lamem_generator(split_file, batch_size):
    while True:  # Infinite loop for batch generation
        random_files = random.sample(split_file, batch_size)  # Random sampling of batch_size elements
        inputs_1 = mp.Pool().map(load_image, [face_images + i[0] for i in random_files])  # Parallel image loading
        final_labels = [[i[1]] for i in random_files]  # Extract labels
        yield (np.array(inputs_1), np.array(final_labels))  # Yield inputs and labels as NumPy arrays

# Prepare train, validation, and test splits
train_split, test_split, valid_split = load_split(mem_score_xlsx)

# Set training parameters
batch_size = 64
train_gen = lamem_generator(train_split, batch_size=batch_size)  # Training generator
valid_gen = lamem_generator(valid_split, batch_size=batch_size)  # Validation generator

# Define training callbacks
callbacks = [
    ModelCheckpoint(filepath="vgg_hf/splitted/epoch_{epoch}.h5"),  # Save model at each epoch
    CSVLogger("vgg_hf/splitted/faces_ft1.log", separator=",", append=False),  # Log training process
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)  # Adjust learning rate when validation loss plateaus
]

# Compile the model with the custom loss function
my_opt = tf.keras.optimizers.Adam(0.0001)  # Use Adam optimizer with low learning rate
model.compile(my_opt, euclidean_distance_loss)  # Compile model for regression

# Train the model using the generator
model.fit_generator(
    train_gen,
    steps_per_epoch=int(len(train_split) / batch_size),  # Number of batches per epoch
    epochs=50,  # Total training epochs
    verbose=1,  # Print progress
    validation_data=valid_gen,
    validation_steps=int(len(test_split) / batch_size),  # Number of validation batches
    callbacks=callbacks  # Use defined callbacks
)
