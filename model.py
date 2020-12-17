import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf
import splitfolders
import cv2
import keras_preprocessing
import random
import os

tf.random.set_seed(1234)

print("--Get data--")

# split Covid and Non-Covid images into training and testing categories
# using 70% for training and 30% for testing/validation
splitfolders.ratio("data", output="data-split", seed=1337, ratio=(.7,.3), group_prefix=None) # default values

X_train_temp = np.zeros((521, 256, 256, 3))
y_train_temp = np.zeros((521, 1))
X_val_temp = np.zeros((225, 256, 256, 3))
y_val_temp = np.zeros((225, 1))

X_train = np.zeros((521, 256, 256, 3))
y_train = np.zeros((521, 1))
X_val = np.zeros((225, 256, 256, 3))
y_val = np.zeros((255, 1))


# Load images
train_covid_path = glob('.\\data-split\\train\\CT_COVID\\*')
train_normal_path = glob('.\\data-split\\train\\CT_NonCOVID\\*')
test_covid_path = glob('.\\data-split\\val\\CT_COVID\\*')
test_normal_path = glob('.\\data-split\\val\\CT_NonCOVID\\*')

#initialize new  arrays

# Building the training array
cnt = 0
for img_file in train_covid_path:
    image_orig = cv2.imread(img_file)
    image_resized = cv2.resize(image_orig, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_array = img_to_array(image_resized)

    X_train_temp[cnt] = img_array/255
    y_train_temp[cnt] = 1
    cnt += 1

for img_file in train_normal_path:
    image_orig = cv2.imread(img_file)
    image_resized = cv2.resize(image_orig, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_array = img_to_array(image_resized)

    X_train_temp[cnt] = img_array/255
    y_train_temp[cnt] = 0
    cnt += 1

# Building the validation array
cnt = 0
for img_file in test_covid_path:
    image_orig = cv2.imread(img_file)
    image_resized = cv2.resize(image_orig, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_array = img_to_array(image_resized)

    X_val_temp[cnt] = img_array/255   # Normalization
    y_val_temp[cnt] = 1
    cnt += 1

for img_file in test_normal_path:
    image_orig = cv2.imread(img_file)
    image_resized = cv2.resize(image_orig, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_array = img_to_array(image_resized)

    X_val_temp[cnt] = img_array/255   # Normalization
    y_val_temp[cnt] = 0
    cnt += 1

# Suffling the train and validation array

num_list1 = [x for x in range(0, 521)]
num_list2 = [x for x in range(0, 225)]

random.shuffle(num_list1)
random.shuffle(num_list2)

for ind, num in enumerate(num_list1):
    X_train[ind] = X_train_temp[num]
    y_train[ind] = y_train_temp[num]

for ind, num in enumerate(num_list2):
    X_val[ind] = X_val_temp[num]
    y_val[ind] = y_val_temp[num]


# Converting an array to the image and displaying it

num = random.randint(0, 746)
temp_img = array_to_img(X_train[num])
plt.imshow(temp_img)
plt.show()
print("Label for the image is(1 is Covid and 0 is normal): ", y_train[num])