import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import numpy as np
import cv2
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import UpSampling2D
import visualkeras

# train_data = np.load('train_data.npy')
# test_data = np.load('test_data.npy')
# train_images = train_data[:, :-12].reshape(-1, 112, 112, 3)
# test_images = test_data[:, :-12].reshape(-1, 112, 112, 3)
# print(np.shape(train_images))
# print(np.shape(train_data[:, :-12]))
# train_images = train_images / 255.0
# test_images = test_images / 255.0


with tf.device('/GPU:0'):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(112, 112, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(12, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

visualkeras.layered_view(model).show()
visualkeras.layered_view(model,to_file='CNN_arc.png')
visualkeras.layered_view(model,to_file='CNN_arc.png').show()
visualkeras.layered_view(model)

# with tf.device('/GPU:0'):
#     model.fit(train_images, train_data[:, -12:], epochs=20, batch_size=32)
# model.save('cnn_model.h5')
#
# test_loss, test_acc = model.evaluate(test_images, test_data[:, -12:])
# print('Test accuracy:', test_acc)
