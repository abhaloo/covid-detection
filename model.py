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


# Load images
train_covid_path = glob('.\\data-split\\train\\CT_COVID\\*')
train_normal_path = glob('.\\data-split\\train\\CT_NonCOVID\\*')
val_covid_path = glob('.\\data-split\\val\\CT_COVID\\*')
val_normal_path = glob('.\\data-split\\val\\CT_NonCOVID\\*')

#initialize new  arrays
X_train_temp = []
Y_train_temp = []
X_val_temp = []
Y_val_temp = []

X_train = []
Y_train = []
X_test = []
Y_test = []


# Building the training array
cnt = 0
for img_file in train_covid_path:
    image_orig = cv2.imread(img_file)
    image_resized = cv2.resize(image_orig, (256, 256))
    img_array = img_to_array(image_resized)
    X_train.append(img_array/255)
    Y_train.append(1)
    cnt += 1

for img_file in train_normal_path:
    image_orig = cv2.imread(img_file)
    image_resized = cv2.resize(image_orig, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_array = img_to_array(image_resized)

    X_train.append(img_array/255)
    Y_train.append(0)
    cnt += 1



# Building the testing array
cnt = 0
for img_file in val_covid_path:
    image_orig = cv2.imread(img_file)
    image_resized = cv2.resize(image_orig, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_array = img_to_array(image_resized)

    X_test.append(img_array / 255)
    Y_test.append(1)
    cnt += 1

for img_file in val_normal_path:
    image_orig = cv2.imread(img_file)
    image_resized = cv2.resize(image_orig, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_array = img_to_array(image_resized)

    X_test.append(img_array / 255)
    Y_test.append(1)
    cnt += 1


# Converting data to numpy array




num = random.randint(0, cnt)
temp_img = array_to_img(X_test[10])
plt.imshow(temp_img)
temp_img.show()
print("Label for the image is(1 is Covid and 0 is normal): ", X_test[10])
