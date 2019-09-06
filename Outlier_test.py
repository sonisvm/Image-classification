import tensorflow as tf
import os
import imageio
import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMAGE_SIZE = 32

model = load_model("svhn_model.h5")

outliers_dir = os.listdir("./Outliers")

images = []
file_name = []

for file in outliers_dir:
    if file.endswith('.png'):
        file_path = "./Outliers/" + file
        image = imageio.imread(file_path)
        image = cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE)).astype('float32')/255.0
        images.append(image)
        file_name.append(file)

images_arr = np.array(images)
images_arr = np.insert(images_arr, 0, 1, 0)

predictions = model.predict(images_arr)

for i in range(0, images_arr.shape[0]-1):
    print("File: " +file_name[i] +" Prediction: " + str(np.argmax(predictions[i])))
