
import cv2 #opencv-python==4.10.0.84
import numpy as np #numpy==1.26.4
import tensorflow as tf #tensorflow==2.17.0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming you have an ImageDataGenerator for the test set
test_datagen = ImageDataGenerator(rescale=1./255)

model = tf.keras.models.load_model('trained_model/face_emotion_tracker_FER2013_50E.h5')

path_to_data = 'data/FER2013/with_augmented/content/data/aug/test'

# Create a generator for the test set
test_generator = test_datagen.flow_from_directory(
  path_to_data,
  target_size=(48, 48),
  # batch_size=64,
  subset="validation"
)

x_batch, y_batch = next(test_generator)
print(f"Input batch shape: {x_batch.shape}")
print(f"Label batch shape: {y_batch.shape}")

# Evaluate the model using the test generator
loss, accuracy = model.evaluate(test_generator, steps=375)  # Adjust steps as needed
print(f"Test Accuracy: {accuracy * 100:.2f}%")

