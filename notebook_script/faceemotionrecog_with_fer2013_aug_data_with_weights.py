

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

from sklearn.utils.class_weight import compute_class_weight

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


print("GPU count ---------------------------------->>> ", gpus)


data_gen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, rescale=1./255, validation_split=0.2)
path_to_data = 'data/FER2013/with_augmented/content/data/aug/train'

training_set = data_gen.flow_from_directory(path_to_data,(48,48),color_mode='grayscale', subset="training")
testing_set = data_gen.flow_from_directory(path_to_data,(48,48),color_mode='grayscale', subset="validation")

print(training_set.class_indices)
print(training_set.batch_size)
labels = ["angry","disgust","fear","happy","neutral","sad","surprise"]

#######################################         ----------------------- MODEL 5 ------------------------------------------
import keras

early = EarlyStopping(monitor='val_loss', patience=5)

face_recognition_model_5 = keras.Sequential()

face_recognition_model_5.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same',kernel_regularizer='l2'))
face_recognition_model_5.add(Conv2D(64, (3, 3), activation='relu',kernel_regularizer='l2'))
face_recognition_model_5.add(BatchNormalization())
face_recognition_model_5.add(MaxPooling2D(pool_size=(2, 2))) #max pooling to decrease dimension
face_recognition_model_5.add(Dropout(0.25)) #test

face_recognition_model_5.add(Conv2D(128, (3, 3), activation='relu',kernel_regularizer='l2'))
face_recognition_model_5.add(BatchNormalization())
face_recognition_model_5.add(MaxPooling2D(pool_size=(2, 2))) #max pooling to decrease dimension
face_recognition_model_5.add(Conv2D(128, (3, 3), activation='relu',kernel_regularizer='l2'))
face_recognition_model_5.add(BatchNormalization())
face_recognition_model_5.add(Dropout(0.25))

face_recognition_model_5.add(Flatten())

face_recognition_model_5.add(Dense(1024, activation = 'relu',kernel_regularizer='l2'))
face_recognition_model_5.add(Dropout(0.5))
face_recognition_model_5.add(Dense(7, activation = 'softmax'))


# compile
face_recognition_model_5.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),loss=keras.losses.CategoricalCrossentropy(),metrics=['acc'])

early = EarlyStopping(monitor='val_loss',patience=10)

face_recognition_model_5.summary()

# Assuming you have the class labels for your training data
# Replace `training_set.classes` with the array of class labels from your dataset
class_labels = training_set.classes  # This contains the class indices for all images

# Get the unique class indices and their corresponding weights
class_weights = compute_class_weight(
    class_weight='balanced',  # Automatically balance weights
    classes=np.unique(class_labels),  # Unique class indices
    y=class_labels  # Class labels for all samples
)

# Convert the result to a dictionary
class_weights_dict = dict(enumerate(class_weights))

print("Class Weights:", class_weights_dict)

history_new = face_recognition_model_5.fit(
    training_set, 
    epochs=1, 
    validation_data=testing_set, 
    batch_size=32,
    steps_per_epoch=training_set.samples // 32,
    validation_steps=testing_set.samples // 32,
    callbacks=[early],
    class_weight=class_weights_dict  
    )

face_recognition_model_5.save('face_recognition_model_5_80E_64BS_4SPE_650VS.h5')

def plot_accuracy(history):
    plt.plot(history.history["acc"], color='red')
    plt.plot(history.history["val_acc"], color='orange')

    plt.title("Accuracy")

    plt.legend(["acc","val_acc"], bbox_to_anchor =(0.65, 1.00))
    plt.show()

def plot_loss(history):
    plt.plot(history.history["loss"], color='blue')
    # plt.plot(history.history["loss"],color='green')
    plt.plot(history.history["val_loss"],color='olive')
    # plt.plot(history.history["recall_1"],color='violet')
    # plt.plot(history.history["specificity_at_sensitivity_1"],color='purple')

    # plt.plot(history.history["val_accuracy"],color='cyan')
    # plt.plot(history.history["val_auc_1"],color='yellow')

    plt.title("Loss")

    plt.legend(["loss", "val_loss" ], bbox_to_anchor =(0.65, 1.00))
    plt.show()

plot_accuracy(history_new)
plot_loss(history_new)