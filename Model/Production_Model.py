import tensorflow as tf
import json
import numpy as np
import os
import pathlib
import pandas
from sklearn import preprocessing
from tensorflow import keras
MODEL_PATH = "./SavedModel/model"

model = tf.keras.models.load_model(MODEL_PATH)

# Display the model's architecture
model.summary()