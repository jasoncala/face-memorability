# Import required libraries
import numpy as np  # Numerical computations
import pandas as pd  # Data manipulation and analysis
import PIL
from PIL import Image  # Image processing
from datetime import datetime, timedelta  # For time-related operations
import os  # For file system operations
from tqdm import tqdm  # For progress bar
import csv  # For working with CSV files
from sklearn.model_selection import train_test_split  # To split data into train/test
import random  # For random sampling
import multiprocessing as mp  # For parallel processing
from keras_vggface.vggface import VGGFace  # Pretrained VGGFace model

# TensorFlow and Keras imports for deep learning
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential

"""
This script fine-tunes the ResNet50 backbone of VGGFace for memorability prediction with 
dropout layers for regularization. It includes a regression output layer and uses 
transfer learning, along with data generators and callbacks for robust training.
"""


# Path to the Excel file containing memorability scores
mem_score_xlsx = "faces/Memorability Scores/memorability-scores.xlsx"

# Path to the directory containing face images
face_images = "faces/10k US Adult Faces Database/Face Images/"

# Define a custom loss function (Euclidean distance loss)
def euclidean_distance_loss(y_true, y_pred):
    """
    Computes the Euclidean distance between the true and predicted values.
    Used as a loss function for regression tasks.
    """
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_pred - y_true), axis=-1))

# Load the pretrained VGGFace model with ResNet50 architecture
model = VGGFace(model='resnet50')
outputs = model.layers[-2].output  # Access the second-to-last layer's output
drop = tf.keras.layers.Dropout(0.5)(outputs)  # Add a dropout layer for regularization
output = Dense(1)(drop)  # Add a dense layer with 1 output for memorability score prediction

# Define the new model with the modified architecture
model = Model(inputs=model.inputs, outputs=output)
print(model.summary())  # Print the model architecture

# Function to freeze certain layers for fine-tuning
def setup_to_finetune(model, N_freeze_layer):
    """
    Freezes the bottom N_freeze_layer layers of the model
    and allows the remaining top layers to be trainable.
    """
    for layer in model.layers[:N_freeze_layer]:
        layer.trainable = False  # Freeze these layers
    for layer in model.layers[N_freeze_layer:]:
        layer.trainable = True  # Unfreeze these layers

# Function to split dataset into training, validation, and test sets
def load_split(split_file):
    """
    Loads the dataset from an Excel file, splits it into train, validation, and test sets,
    and adjusts the scores to subtract the False Alarm Rate (FAR).
    """
    faces_ds = pd.read_excel(split_file)  # Load dataset from Excel
    X_train, X_test = train_test_split(faces_ds, test_size=0.2, random_state=42)  # 80-20 split
    # Extract filenames and scores
    scores_train = list(X_train['Hit Rate (HR)'])
    names_train = list(X_train['Filename'])
    scores_test = list(X_test['Hit Rate (HR)'])
    names_test = list(X_test['Filename'])
    # Adjust scores using False Alarm Rate
    FA_test = list(X_test['False Alarm Rate (FAR)'])
    FA_train = list(X_train['False Alarm Rate (FAR)'])
    X_train = [[names_train[i], scores_train[i] - FA_train[i]] for i in range(len(scores_train))]
    X_test = [[names_test[i], scores_test[i] - FA_test[i]] for i in range(len(scores_test))]
    # Split test set into validation and test subsets
    X_valid = X_test[:int(len(X_test) / 2)]
    X_test = X_test[int(len(X_test) / 2):]
    return X_train, X_test, X_valid

# Function to load and preprocess a single image
def load_image(image_file):
    """
    Loads an image file, resizes it to 224x224 pixels, and applies optional random mirroring.
    """
    image = Image.open(image_file).resize((224, 224)).convert("RGB")  # Resize to 224x224 and convert to RGB
    if random.uniform(0, 1) > 0.5:  # Apply random mirroring
        return np.array(image.transpose(PIL.Image.FLIP_LEFT_RIGHT), dtype=np.uint8)
    else:
        return np.array(image, dtype=np.uint8)

# Function to preprocess an image for IRv2 input size (299x299)
def load_image_IRv2(image_file):
    """
    Loads an image file and resizes it to 299x299 pixels for IRv2 compatibility.
    """
    image = Image.open(image_file).resize((299, 299)).convert("RGB")
    return np.array(image, dtype=np.uint8)

# Generator to create batches of images and labels
def lamem_generator(split_file, batch_size):
    """
    Generates batches of image data and corresponding labels for training or evaluation.
    """
    while True:  # Infinite loop for generator
        random_files = random.sample(split_file, batch_size)  # Randomly sample a batch
        # Load images in parallel using multiprocessing
        inputs_1 = mp.Pool().map(load_image, [face_images + i[0] for i in random_files])
        final_labels = [[i[1]] for i in random_files]  # Extract labels
        yield (np.array(inputs_1), np.array(final_labels))  # Yield inputs and labels as NumPy arrays

# Prepare train, validation, and test splits
train_split, test_split, valid_split = load_split(mem_score_xlsx)

# Define batch size for training
batch_size = 64

# Create data generators for training and validation
train_gen = lamem_generator(train_split, batch_size=batch_size)
valid_gen = lamem_generator(valid_split, batch_size=batch_size)

# Define training callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath="res_hf/drop/epoch_{epoch}.h5"),  # Save model at each epoch
    tf.keras.callbacks.CSVLogger("res_hf/drop/faces_ft1.log", separator=",", append=False),  # Log training metrics
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)  # Reduce LR on plateau
]

# Compile the model with Adam optimizer and custom loss function
my_opt = tf.keras.optimizers.Adam(0.0001)  # Adam optimizer with low learning rate
model.compile(my_opt, euclidean_distance_loss)

# Train the model using the data generators
model.fit_generator(
    train_gen,
    steps_per_epoch=int(len(train_split) / batch_size),  # Number of batches per epoch
    epochs=50,  # Total number of epochs
    verbose=1,  # Print progress
    validation_data=valid_gen,
    validation_steps=int(len(test_split) / batch_size),  # Number of validation batches
    callbacks=callbacks  # Use defined callbacks
)
