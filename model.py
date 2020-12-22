# Importing necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.backend import clear_session
from keras_preprocessing.image import ImageDataGenerator, np
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from random import random
import splitfolders
import cv2
import random
import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)


# Image will be rotated by upto 365 degrees. The image will also be shifting
# left and right by 5 pixels. This will generate a bigger dataset to run
# CNN on. Possible to generate 146000 images from one image.
def imageAugmentation(image):
    seed = int(random() * 100000000)
    wShift = 5
    hShift = 5
    # rotation is changing values of image
    imgen = ImageDataGenerator(
        rotation_range=365,
        width_shift_range=wShift,
        height_shift_range=hShift,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant',
        cval=0)
    image = imgen.random_transform(image, seed)
    return image

print("--Get data--")

# split Covid and Non-Covid images into training and testing categories
# using 70% for training (521 files) and 30% for testing/validation (225 files)
splitfolders.ratio("data", output="data-split", seed=1337, ratio=(.7, .3), group_prefix=None)  # default values

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

# Load images (For google collab)
# train_covid_path = glob('/content/data-split/train/CT_COVID/*')
# train_normal_path = glob('/content/data-split/train/CT_NonCOVID/*')
# test_covid_path = glob('/content/data-split/val/CT_COVID/*')
# test_normal_path = glob('/content/data-split/val/CT_NonCOVID/*')

# Building the training array
cnt = 0  # index
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
cnt2 = 0  # index
for img_file in test_covid_path:
    image_orig = cv2.imread(img_file)
    image_resized = cv2.resize(image_orig, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_array = img_to_array(image_resized)

    x_test_temp[cnt2] = img_array / 255  # Normalization
    y_test_temp[cnt2] = 1
    cnt2 += 1

for img_file in test_normal_path:
    image_orig = cv2.imread(img_file)
    image_resized = cv2.resize(image_orig, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_array = img_to_array(image_resized)

    x_test_temp[cnt2] = img_array / 255  # Normalization
    y_test_temp[cnt2] = 0
    cnt += 1

# Shuffling the train and validation array

# initialize arrays for after data shuffle
x_train = np.zeros((521, 256, 256, 3))
y_train = np.zeros((521, 1))
x_test = np.zeros((225, 256, 256, 3))
y_test = np.zeros((225, 1))

# create indices arrays
train_indices = [x for x in range(0, 521)]
test_indicies = [x for x in range(0, 225)]

# shuffle indicies
random.shuffle(train_indices)
random.shuffle(test_indicies)

# populate arrays
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
# plt.show() #if this is uncommented you must close the window displaying the image to continue program execution beyond this point


# Build CNN model
def build_model(input_shape, classes=1):
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_conv1')(x)
    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv1')(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv2')(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv1')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv1')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='sigmoid', name='predictions')(x)
    model = Model(img_input, x, name="vgg16_Covid")

    return model


clear_session()
model = build_model(x_train.shape[1:])
opt = tf.keras.optimizers.Adam(lr=0.01, decay=1e-6)  # changed from adam optimizer to sgd with learning rate decay
sgd = tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

print("-- Fit model--")
model.fit(x_train, y_train, batch_size=16, epochs=20, verbose=1, use_multiprocessing=False,
          steps_per_epoch=50, validation_steps=50 * 0.3, validation_data=(x_test, y_test))

# Model performance evaluation

print("-- Evaluate model--")

model_loss, model_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuracy: {model_acc * 100:.1f}%")
