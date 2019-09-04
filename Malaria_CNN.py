# Classifying Malaria data set
import tensorflow as tf
import numpy as np
import os
import cv2
import time
import imageio
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Constants
BATCH_SIZE = 100
EPOCHS = 10
CLASSES = 2
IMAGE_SIZE = 32

# Get the dataset
parasitized = os.listdir("./cell_images/Parasitized")
uninfected = os.listdir("./cell_images/Uninfected")

cells = []
labels = []

for file in parasitized:
    if file.endswith('.png'):
        file_path = "./cell_images/Parasitized/" + file
        image = imageio.imread(file_path)
        image = cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE)).astype('float32')/255.0
        cells.append(image)
        labels.append(1)

for file in uninfected:
    if file.endswith('.png'):
        file_path = "./cell_images/Uninfected/" + file
        image = imageio.imread(file_path)
        image = cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE)).astype('float32')/255.0
        cells.append(image)
        labels.append(0)

print("Number of samples: ", len(cells))

# Reorder the data

cells_array = np.array(cells[0:27000])
labels_array = np.array(labels[0:27000])
cells.clear()
labels.clear()

s=np.arange(cells_array.shape[0])
np.random.shuffle(s)
reordered_cells=cells_array[s]
reordered_labels=labels_array[s]
cells_array=[]
labels_array=[]

# Split into training and test
train_data, test_data, train_label, test_label = train_test_split(reordered_cells, reordered_labels,
                                                train_size=0.80, random_state=111)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same',input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu', padding='same'),
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
history = model.fit(
    train_data, train_label,
    steps_per_epoch = int(np.ceil(len(train_data)/float(BATCH_SIZE))),
    epochs=EPOCHS)
end = time.time()

model.save("malaria_model.h5")
#!ls -l --block-size=M

print("Training accuracy: ", history.history['accuracy'][-1])
print("Training time: ", end-start)

start = time.time()
# Evaluate the model
scores = model.evaluate(test_data, test_label, steps = int(np.ceil(len(test_data) / float(BATCH_SIZE))))
end = time.time()

print("Testing accuracy: ", scores[1])
print("Testing time: ", end-start)
