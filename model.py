# Importing necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,MaxPooling2D
from tensorflow.keras.models import Model
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import splitfolders
import cv2
import random
import tensorflow as tf

tf.random.set_seed(1234)

print("--Get data--")

# split Covid and Non-Covid images into training and testing categories
# using 70% for training (521 files) and 30% for testing/validation (225 files)
splitfolders.ratio("data", output="data-split", seed=1337, ratio=(.7,.3), group_prefix=None) # default values

# np.zeroes arguments: number of files, len, width, color_dimension
# initialize training and testing numpy arrays for loading data
x_train_temp = np.zeros((521, 256, 256, 3))
y_train_temp = np.zeros((521, 1))
x_test_temp = np.zeros((225, 256, 256, 3))
y_test_temp = np.zeros((225, 1))


# Load images
train_covid_path = glob('.\\data-split\\train\\CT_COVID\\*')
train_normal_path = glob('.\\data-split\\train\\CT_NonCOVID\\*')
test_covid_path = glob('.\\data-split\\val\\CT_COVID\\*')
test_normal_path = glob('.\\data-split\\val\\CT_NonCOVID\\*')

# Building the training array
cnt = 0 #index
for img_file in train_covid_path:
    image_orig = cv2.imread(img_file)
    image_resized = cv2.resize(image_orig, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_array = img_to_array(image_resized)

    x_train_temp[cnt] = img_array / 255
    y_train_temp[cnt] = 1
    cnt += 1

for img_file in train_normal_path:
    image_orig = cv2.imread(img_file)
    image_resized = cv2.resize(image_orig, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_array = img_to_array(image_resized)

    x_train_temp[cnt] = img_array / 255
    y_train_temp[cnt] = 0
    cnt += 1

# Building the validation array
cnt2 = 0 #index
for img_file in test_covid_path:
    image_orig = cv2.imread(img_file)
    image_resized = cv2.resize(image_orig, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_array = img_to_array(image_resized)

    x_test_temp[cnt2] = img_array / 255   # Normalization
    y_test_temp[cnt2] = 1
    cnt2 += 1

for img_file in test_normal_path:
    image_orig = cv2.imread(img_file)
    image_resized = cv2.resize(image_orig, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_array = img_to_array(image_resized)

    x_test_temp[cnt2] = img_array / 255   # Normalization
    y_test_temp[cnt2] = 0
    cnt += 1


# Shuffling the train and validation array

#initialize arrays for after data shuffle
x_train = np.zeros((521, 256, 256, 3))
y_train = np.zeros((521, 1))
x_test = np.zeros((225, 256, 256, 3))
y_test = np.zeros((225, 1))

#create indices arrays
train_indices = [x for x in range(0, 521)]
test_indicies = [x for x in range(0, 225)]

#shuffle indicies
random.shuffle(train_indices)
random.shuffle(test_indicies)

#populate arrays
for ind, num in enumerate(train_indices):
    x_train[ind] = x_train_temp[num]
    y_train[ind] = y_train_temp[num]

for ind, num in enumerate(test_indicies):
    x_test[ind] = x_test_temp[num]
    y_test[ind] = y_test_temp[num]


# Test to display random image
#
# print("--Display image--")
# num = random.randint(0, 746)
# temp_img = array_to_img(X_train[num])
# plt.imshow(temp_img)
#plt.show() #if this is uncommented you must close the window displaying the image to continue program execution beyond this point


# Build CNN model
def build_model(input_shape):
    X_train = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_train)
    X = Conv2D(8, (7, 7), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)

    X = Flatten()(X)
    X = Dense(20, activation='relu')(X)  # Amitabh added this dense layer
    X = Dense(1, activation='sigmoid')(X)

    model = Model(inputs=X_train, outputs=X)

    return model


model = build_model(x_train.shape[1:])
opt = tf.keras.optimizers.Adam(lr=0.1, decay=1e-6) #changed from adam optimizer to sgd with learning rate decay
model.compile(optimizer = opt, loss ='binary_crossentropy', metrics = ['accuracy'])

print("-- Fit model--")
model.fit(x_train, y_train, batch_size = 100, epochs = 5, verbose = 2)

# Model performance evaluation

print("-- Evaluate model--")

model_loss, model_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuracy: {model_acc*100:.1f}%")