

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import matplotlib as plt
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Reshape, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
import albumentations as alb
import time
import cv2
from tensorflow.keras.applications import ResNet152V2, VGG16

from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam




# Define constants
IMG_HEIGHT = 48
IMG_WIDTH = 48
NUM_CLASSES = 7
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.0001
path_to_data = 'data/FER2013/with_augmented/content/data/aug/train'

# Load the VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling
x = Dense(256, activation='relu')(x)  # Fully connected layer
x = Dropout(0.5)(x)  # Dropout for regularization
output = Dense(NUM_CLASSES, activation='softmax')(x)  # Output layer for 7 classes

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    # rescale=1.0/255,
    # rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # fill_mode='nearest',
    validation_split=0.2  # Split for training and validation
)

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    path_to_data,  # Replace with the path to your dataset
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    path_to_data,  # Replace with the path to your dataset
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS
)

# Fine-tuning (optional)
# Unfreeze some layers of the base model for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:15]:  # Freeze the first 15 layers
    layer.trainable = False

# Recompile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE / 10),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tune the model
fine_tune_epochs = 10
total_epochs = EPOCHS + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1]
)

# Save the model
model.save('vgg16_7_class_model.h5')

