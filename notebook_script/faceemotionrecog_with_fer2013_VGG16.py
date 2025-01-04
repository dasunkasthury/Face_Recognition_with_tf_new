

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import matplotlib as plt
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Reshape, Dropout, BatchNormalization
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

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

print("GPU count ---------------------------------->>> ", gpus)

data_gen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, rescale=1./255, validation_split=0.2)
path_to_data = 'data/FER2013/with_augmented/content/content/data/data_with_accurate_data/aug/train'
# path_to_data = 'data\FER2013\with_augmented\content\content\data\data_with_accurate_data\aug\train'  
training_set = data_gen.flow_from_directory(path_to_data,(48,48),color_mode='grayscale', subset="training")
testing_set = data_gen.flow_from_directory(path_to_data,(48,48),color_mode='grayscale', subset="validation")

print(training_set.class_indices)
print(training_set.batch_size)
labels = ["angry","disgust","fear","happy","neutral","sad","surprise"]

vgg = VGG16(include_top=False) # originally it used ssd the move to vgg16

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Modify the input layer to accept grayscale images
input_layer = Input(shape=(48, 48, 1))
x = tf.keras.layers.Concatenate()([input_layer, input_layer, input_layer])  # Convert 1 channel to 3

# Pass the input through the base model
x = base_model(x, training=False)

# Add custom layers on top
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(8, activation='relu')(x)
output_layer = Dense(7, activation='softmax')(x)

# Create the model
face_recognition_model_vgg16 = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
face_recognition_model_vgg16.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
face_recognition_model_vgg16.summary()

early = EarlyStopping(monitor='val_loss', patience=5)

face_recognition_model_vgg16 = face_recognition_model_vgg16.fit(
    training_set,
    epochs=10,
    validation_data=testing_set,
    batch_size=32,
    steps_per_epoch=700,
    validation_steps=200, #validation_samples // batch_size
    callbacks=[early]
    )

# face_recognition_model_vgg16.save('face_recognition_model_5_80E_64BS_4SPE_650VS.h5')

# def plot_accuracy(history):
#     plt.plot(history.history["acc"], color='red')
#     plt.plot(history.history["val_acc"], color='orange')

#     plt.title("Accuracy")

#     plt.legend(["acc","val_acc"], bbox_to_anchor =(0.65, 1.00))
#     plt.show()

# def plot_loss(history):
#     plt.plot(history.history["loss"], color='blue')
#     # plt.plot(history.history["loss"],color='green')
#     plt.plot(history.history["val_loss"],color='olive')
#     # plt.plot(history.history["recall_1"],color='violet')
#     # plt.plot(history.history["specificity_at_sensitivity_1"],color='purple')

#     # plt.plot(history.history["val_accuracy"],color='cyan')
#     # plt.plot(history.history["val_auc_1"],color='yellow')

#     plt.title("Loss")

#     plt.legend(["loss", "val_loss" ], bbox_to_anchor =(0.65, 1.00))
#     plt.show()

# plot_accuracy(history_new)
# plot_loss(history_new)