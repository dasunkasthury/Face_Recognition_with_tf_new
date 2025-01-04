import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

 
  # Set to '0' for full logs