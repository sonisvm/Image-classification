# Classifying Flowers data set
import tensorflow as tf
import numpy as np
import os
import cv2
import time
import imageio
import scipy.io as sio
from tensorflow.keras.datasets import cifar100
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Constants
BATCH_SIZE = 100
EPOCHS = 100
CLASSES = 100
IMAGE_SIZE = 32

(train_data, train_labels), (test_data, test_labels) = cifar100.load_data()

print("Number of training samples: ", len(train_data))
print("Number of testing samples: ", len(test_data))

train_data_generator = ImageDataGenerator(rescale=1./255,rotation_range=40,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True,
                                            fill_mode='nearest')
test_data_generator = ImageDataGenerator(rescale=1./255)

train_labels = tf.keras.utils.to_categorical(train_labels, CLASSES)
test_labels = tf.keras.utils.to_categorical(test_labels, CLASSES)

# Configure data generators
train_data_gen = train_data_generator.flow(x=train_data, y=train_labels,
                                            batch_size = BATCH_SIZE, shuffle=True)
test_data_gen = test_data_generator.flow(x=test_data, y=test_labels,
                                            batch_size=BATCH_SIZE)
print("Setting up layers")
# Setup the layers
model = Sequential([
    Conv2D(128, (3,3), activation='elu', padding='same', input_shape=(32,32,3)),
    MaxPooling2D(2, 2),

    Conv2D(128, (3,3), activation='elu', padding='same'),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='elu', padding='same'),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='elu', padding='same'),
    MaxPooling2D(2,2),


    Flatten(),
    Dense(128, activation='elu'),
    Dropout(0.25),
    Dense(256, activation='elu'),
    Dropout(0.25),
    Dense(512, activation='elu'),
    Dropout(0.25),
    Dense(1024, activation='elu'),
    Dropout(0.25),
    Dense(CLASSES, activation='softmax')
])

print("Compiling")
# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

model.summary()

print("Fitting the model")

start=time.time()
# Fit the model
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(len(train_data) / float(BATCH_SIZE))),
    epochs=EPOCHS)
end = time.time()

model.save("cifar100_model.h5")
!ls -l --block-size=M

print("Training accuracy: ", history.history['accuracy'][-1])
print("Training time: ", end-start)

start = time.time()
# Evaluate the model
scores = model.evaluate_generator(test_data_gen, steps = int(np.ceil(len(test_data) / float(BATCH_SIZE))))
end = time.time()

print("Testing accuracy: ", scores[1])
print("Testing time: ", end-start)
