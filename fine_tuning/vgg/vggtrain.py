# Import libraries (same as `vgg_hf.py`)
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import csv
from sklearn.model_selection import train_test_split
import random
import multiprocessing as mp
from keras_vggface.vggface import VGGFace
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

"""
This script fine-tunes the VGGFace model for memorability prediction with adjusted 
configurations for logging and learning rate schedules. It focuses on training the model 
while optimizing for different experimental conditions.
"""


# Paths for memorability scores and face images
mem_score_xlsx = "faces/Memorability Scores/memorability-scores.xlsx"
face_images = "faces/10k US Adult Faces Database/Face Images/"

# Custom loss function for regression
def euclidean_distance_loss(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_pred - y_true), axis=-1))

# Load and modify VGGFace model
model = VGGFace(model='vgg16')  # Use VGG16 architecture
outputs = model.layers[-2].output  # Second-to-last layer
output = Dense(1)(outputs)  # Add dense layer for regression
model = Model(inputs=model.inputs, outputs=output)  # Create new model
print(model.summary())

# Fine-tuning setup
def setup_to_finetune(model, N_freeze_layer):
    for layer in model.layers[:N_freeze_layer]:
        layer.trainable = False  # Freeze these layers
    for layer in model.layers[N_freeze_layer:]:
        layer.trainable = True  # Allow training on remaining layers

# Dataset split function (same as `vgg_hf.py`)
def load_split(split_file):
    faces_ds = pd.read_excel(split_file)
    X_train, X_test = train_test_split(faces_ds, test_size=0.2, random_state=42)
    scores_train = list(X_train['Hit Rate (HR)'] - X_train['False Alarm Rate (FAR)'])
    names_train = list(X_train['Filename'])
    scores_test = list(X_test['Hit Rate (HR)'] - X_test['False Alarm Rate (FAR)'])
    names_test = list(X_test['Filename'])
    X_train = [[names_train[i], scores_train[i]] for i in range(len(scores_train))]
    X_test = [[names_test[i], scores_test[i]] for i in range(len(scores_test))]
    X_valid = X_test[:int(len(X_test)/2)]
    X_test = X_test[int(len(X_test)/2):]
    return X_train, X_test, X_valid

# Image preprocessing function (same as `vgg_hf.py`)
def load_image(image_file):
    image = Image.open(image_file).resize((224,224)).convert("RGB")
    return np.array(image, dtype=np.uint8)

# Data generator (same as `vgg_hf.py`)
def lamem_generator(split_file, batch_size):
    while True:
        random_files = random.sample(split_file, batch_size)
        inputs_1 = mp.Pool().map(load_image, [face_images + i[0] for i in random_files])
        final_labels = [[i[1]] for i in random_files]
        yield (np.array(inputs_1), np.array(final_labels))

# Prepare data splits
train_split, test_split, valid_split = load_split(mem_score_xlsx)

# Training parameters
batch_size = 64
train_gen = lamem_generator(train_split, batch_size=batch_size)
valid_gen = lamem_generator(valid_split, batch_size=batch_size)

# Define callbacks (slightly different paths)
callbacks = [
    ModelCheckpoint(filepath="vgg_hf/final/epoch_{epoch}.h5"),  # Save model for final evaluation
    CSVLogger("vgg_hf/final/faces_ft.log", separator=",", append=False),  # Save logs for this version
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)  # Reduce LR with different factor
]

# Compile model
my_opt = tf.keras.optimizers.Adam(0.0001)  # Adam optimizer
model.compile(my_opt, euclidean_distance_loss)

# Train model
model.fit_generator(
    train_gen,
    steps_per_epoch=int(len(train_split) / batch_size),
    epochs=50,
    verbose=1,
    validation_data=valid_gen,
    validation_steps=int(len(valid_split) / batch_size),
    callbacks=callbacks
)
