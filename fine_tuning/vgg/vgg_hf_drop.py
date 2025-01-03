# Import required libraries
import numpy as np
import pandas as pd
import PIL
from PIL import Image  # Image processing
from datetime import datetime, timedelta
import os
from tqdm import tqdm  # Progress bar
import csv
from sklearn.model_selection import train_test_split  # Dataset splitting
import random
import multiprocessing as mp  # Parallel processing
from keras_vggface.vggface import VGGFace  # Pretrained VGGFace model
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

"""
This script fine-tunes a VGGFace model with dropout layers for memorability score prediction.
It uses transfer learning by adding a dense regression output layer and employs a custom 
Euclidean distance loss function. Includes data generators, callbacks for saving models, 
logging metrics, and reducing learning rates on plateau.
"""

# Define paths for memorability scores and face images
mem_score_xlsx = "faces/Memorability Scores/memorability-scores.xlsx"
face_images = "faces/10k US Adult Faces Database/Face Images/"

# Custom loss function (Euclidean distance loss)
def euclidean_distance_loss(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_pred - y_true), axis=-1))

# Load VGGFace model
model = VGGFace(model='vgg16')
outputs = model.layers[-2].output  # Get the second-to-last layer
drop = tf.keras.layers.Dropout(0.5)(outputs)  # Add dropout for regularization
output = Dense(1)(drop)  # Add dense layer for memorability prediction
model = Model(inputs=model.inputs, outputs=output)  # Define new model
print(model.summary())  # Print model summary

# Function to freeze layers for fine-tuning
def setup_to_finetune(model, N_freeze_layer):
    """Freeze the bottom N_freeze_layers and retrain the remaining layers."""
    for layer in model.layers[:N_freeze_layer]:
        layer.trainable = False
    for layer in model.layers[N_freeze_layer:]:
        layer.trainable = True

# Function to split dataset into train, validation, and test sets
def load_split(split_file):
    faces_ds = pd.read_excel(split_file)  # Load data from Excel
    X_train, X_test = train_test_split(faces_ds, test_size=0.2, random_state=42)
    # Adjust scores by subtracting false alarm rates
    scores_train = list(X_train['Hit Rate (HR)'] - X_train['False Alarm Rate (FAR)'])
    names_train = list(X_train['Filename'])
    scores_test = list(X_test['Hit Rate (HR)'] - X_test['False Alarm Rate (FAR)'])
    names_test = list(X_test['Filename'])
    X_train = [[names_train[i], scores_train[i]] for i in range(len(scores_train))]
    X_test = [[names_test[i], scores_test[i]] for i in range(len(scores_test))]
    X_valid = X_test[:int(len(X_test)/2)]  # Validation set
    X_test = X_test[int(len(X_test)/2):]  # Test set
    return X_train, X_test, X_valid

# Function to preprocess and load a single image
def load_image(image_file):
    image = Image.open(image_file).resize((224,224)).convert("RGB")  # Resize to 224x224
    if random.uniform(0, 1) > 0.5:  # Random mirroring
        return np.array(image.transpose(PIL.Image.FLIP_LEFT_RIGHT), dtype=np.uint8)
    else:
        return np.array(image, dtype=np.uint8)

# Data generator for training and validation
def lamem_generator(split_file, batch_size):
    while True:  # Infinite generator loop
        random_files = random.sample(split_file, batch_size)  # Random batch
        inputs_1 = mp.Pool().map(load_image, [face_images + i[0] for i in random_files])
        final_labels = [[i[1]] for i in random_files]
        yield (np.array(inputs_1), np.array(final_labels))

# Prepare data splits
train_split, test_split, valid_split = load_split(mem_score_xlsx)

# Training parameters
batch_size = 64
train_gen = lamem_generator(train_split, batch_size=batch_size)
valid_gen = lamem_generator(valid_split, batch_size=batch_size)

# Callbacks for training
callbacks = [
    ModelCheckpoint(filepath="vgg_hf/drop/epoch_{epoch}.h5"),  # Save model at each epoch
    CSVLogger("vgg_hf/drop/faces_ft1.log", separator=",", append=False),  # Log training process
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)  # Adjust learning rate
]

# Compile and train the model
my_opt = tf.keras.optimizers.Adam(0.0001)  # Define optimizer
model.compile(my_opt, euclidean_distance_loss)  # Compile model
model.fit_generator(
    train_gen,
    steps_per_epoch=int(len(train_split) / batch_size),
    epochs=50,
    verbose=1,
    validation_data=valid_gen,
    validation_steps=int(len(test_split) / batch_size),
    callbacks=callbacks
)
