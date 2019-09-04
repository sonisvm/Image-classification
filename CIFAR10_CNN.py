# Classifying Cifar10
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from skimage.transform import resize

# Constants
BATCH_SIZE = 100
EPOCHS = 30
CLASSES = 10

# Get the data
(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

print("Initializing generators")
# Initialize image generators
train_data_generator = ImageDataGenerator(rescale=1./255)
test_data_generator = ImageDataGenerator(rescale=1./255)

# Configure data generators
train_data_gen = train_data_generator.flow(x=train_data, y=train_labels,
                                            batch_size = BATCH_SIZE, shuffle=True)
test_data_gen = test_data_generator.flow(x=test_data, y=test_labels,
                                            batch_size=BATCH_SIZE)
print("Setting up layers")
# Setup the layers
model = Sequential([
    Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    MaxPooling2D(2, 2),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),

    Conv2D(512, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(256, activation='relu'),
    Dropout(0.25),
    Dense(512, activation='relu'),
    Dropout(0.25),
    Dense(1024, activation='relu'),
    Dropout(0.25),
    Dense(CLASSES, activation='softmax')
])

print("Compiling")
# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
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

model.save("cifar_model.h5")
#!ls -l --block-size=M

print("Training accuracy: ", history.history['accuracy'][-1])
print("Training time: ", end-start)

start = time.time()
# Evaluate the model
scores = model.evaluate_generator(test_data_gen, steps = int(np.ceil(len(test_data) / float(BATCH_SIZE))))
end = time.time()

print("Testing accuracy: ", scores[1])
print("Testing time: ", end-start)
