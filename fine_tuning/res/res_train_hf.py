# Import required libraries
import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation and analysis
import PIL
from PIL import Image  # For image processing
from datetime import datetime, timedelta  # For handling dates and time
import os  # For file and directory operations
from tqdm import tqdm  # For progress bar functionality
import csv  # For handling CSV files
from sklearn.model_selection import train_test_split  # For splitting datasets into train/test
import random  # For random sampling
import multiprocessing as mp  # For parallel processing
from keras_vggface.vggface import VGGFace  # Pretrained VGGFace model

# TensorFlow and Keras for deep learning
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import metrics

"""
This script fine-tunes the ResNet50 backbone of VGGFace for memorability prediction. 
It does not include dropout layers and focuses on standard fine-tuning configurations. 
The script is optimized for regression tasks and employs a custom Euclidean distance loss.
"""


# Path to the Excel file containing memorability scores
mem_score_xlsx = "faces/Memorability Scores/memorability-scores.xlsx"

# Path to the directory containing face images
face_images = "faces/10k US Adult Faces Database/Face Images/"

# Define a custom loss function (Euclidean distance loss)
def euclidean_distance_loss(y_true, y_pred):
    """
    Computes the Euclidean distance between true and predicted values.
    Used as the loss function for regression tasks.
    """
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_pred - y_true), axis=-1))

# Load the pretrained VGGFace model with ResNet50 architecture
model = VGGFace(model='resnet50')  # Use the ResNet50 backbone of VGGFace
outputs = model.layers[-2].output  # Access the second-to-last layer
output = Dense(1)(outputs)  # Add a dense layer for memorability score prediction

# Define a new model with the modified architecture
model = Model(inputs=model.inputs, outputs=output)
print(model.summary())  # Print the architecture of the model

# Function to set up fine-tuning by freezing layers
def setup_to_finetune(model, N_freeze_layer):
    """
    Freezes the bottom N_freeze_layers of the model for transfer learning.
    The remaining layers will be trainable.
    """
    for layer in model.layers[:N_freeze_layer]:
        layer.trainable = False  # Freeze these layers
    for layer in model.layers[N_freeze_layer:]:
        layer.trainable = True  # Keep these layers trainable

# Function to split the dataset into training, validation, and test sets
def load_split(split_file):
    """
    Reads the dataset from an Excel file, splits it into train/test, 
    and adjusts the scores using False Alarm Rate (FAR).
    """
    faces_ds = pd.read_excel(split_file)  # Load the dataset
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
    Loads an image file and resizes it to 224x224 pixels.
    """
    image = Image.open(image_file).resize((224, 224)).convert("RGB")  # Resize and convert to RGB
    return np.array(image, dtype=np.uint8)  # Return as NumPy array

# Generator to yield batches of images and labels
def lamem_generator(split_file, batch_size):
    """
    Generates batches of images and their corresponding labels.
    """
    while True:  # Infinite loop to keep generating batches
        random_files = random.sample(split_file, batch_size)  # Randomly sample a batch
        # Load images in parallel using multiprocessing
        inputs_1 = mp.Pool().map(load_image, [face_images + i[0] for i in random_files])
        final_labels = [[i[1]] for i in random_files]  # Extract labels
        yield (np.array(inputs_1), np.array(final_labels))  # Yield batch as NumPy arrays

# Prepare train, validation, and test splits
train_split, test_split, valid_split = load_split(mem_score_xlsx)

# Define batch size for training
batch_size = 64

# Create data generators for training and validation
train_gen = lamem_generator(train_split, batch_size=batch_size)
valid_gen = lamem_generator(valid_split, batch_size=batch_size)

# Define callbacks for training
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath="res_hf/splitted/epoch_{epoch}.h5"),  # Save model checkpoints
    tf.keras.callbacks.CSVLogger("res_hf/splitted/faces_ft.log", separator=",", append=False),  # Log training metrics
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)  # Reduce LR on plateau
]

# Compile the model with the Adam optimizer and custom loss function
my_opt = tf.keras.optimizers.Adam(0.0001)  # Adam optimizer with a low learning rate
model.compile(my_opt, euclidean_distance_loss)

# Train the model using the data generators
model.fit_generator(
    train_gen,
    steps_per_epoch=int(len(train_split) / batch_size),  # Number of training steps per epoch
    epochs=50,  # Total number of epochs
    verbose=1,  # Print progress during training
    validation_data=valid_gen,  # Validation data generator
    validation_steps=int(len(valid_split) / batch_size),  # Number of validation steps
    callbacks=callbacks  # Use defined callbacks
)
