import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import numpy as np
import cv2
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D,MaxPooling2D, Flatten, Dense, Dropout, Activation, Conv2DTranspose
from keras.layers import BatchNormalization, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator

# Load the training and test data
train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')

# Split the data into images and targets
train_images = train_data[:, :-12].reshape(-1, 224, 224, 3)
test_images = test_data[:, :-12].reshape(-1, 224, 224, 3)

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

train_X = train_images / 255.0
test_X = test_images / 255.0
input_shape = (224, 224, 3)
num_classes = 12

# Get the targets
target_data = np.expand_dims(train_data[:, -12:], axis=(1, 2))
test_y = np.expand_dims(test_data[:, -12:], axis=(1, 2))
train_y = np.tile(target_data, (1, 224, 224, 1))
test_y = np.tile(test_y, (1, 224, 224, 1))

# Define the data generator for data augmentation
with tf.device('/GPU:0'):
    model = Sequential()
    # Encoder
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D((2, 2)))  # 56x56x32


    model.add(Conv2D(128, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.6))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.6))
    model.add(MaxPooling2D((2, 2)))  # 28x28x256

    model.add(Conv2D(512, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(AveragePooling2D((2, 2)))  #14x14x256

    model.add(Conv2D(256, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(AveragePooling2D((2, 2)))  # 14x14x256

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D((2, 2)))  # 7x7x256

    model.add(Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same'))  # 14x14x512
    model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'))  # 14x14x512
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))  # 28x28x256
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same'))  # 56x56x128
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same'))  # 112x112x64
    model.add(Activation('relu'))
    model.add(Conv2D(num_classes, (1, 1), activation='softmax', padding='valid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#train_datagen = ImageDataGenerator()

# Fit the generator on your training set
#train_datagen.fit(train_X)

# Define the learning rate schedule
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 20:
        lr *= 0.5
    if epoch > 30:
        lr *= 0.5
    return lr

# Define the batch size
BATCH_SIZE = 32

# Define a custom data generator to load data in batches
def custom_data_generator(data, batch_size=BATCH_SIZE):
    num_samples = len(data)
    while True:
        # Shuffle the data
        np.random.shuffle(data)
        for offset in range(0, num_samples, batch_size):
            # Get a batch of data
            batch_samples = data[offset:offset+batch_size]

            # Split the batch into images and targets
            images = batch_samples[:, :-12].reshape(-1, 224, 224, 3)
            targets = np.expand_dims(batch_samples[:, -12:], axis=(1, 2))
            targets = np.tile(targets, (1, 224, 224, 1))

            # Normalize the images
            images = images / 255.0

            yield images, targets

# Load the training and test data
train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')

# Define the data generator for data augmentation
train_datagen = ImageDataGenerator()

# Get the number of training steps per epoch
num_train_steps = int(np.ceil(len(train_data) / BATCH_SIZE))

# Get the number of validation steps per epoch
num_val_steps = int(np.ceil(len(test_data) / BATCH_SIZE))

# Train the model using the custom data generator
model.fit_generator(generator=custom_data_generator(train_data),
                    steps_per_epoch=num_train_steps,
                    epochs=30,
                    validation_data=(test_X, test_y),
                    validation_steps=num_val_steps)

# Train the model with the augmented data generator
with tf.device('/GPU:0'):
    model.fit(train_datagen.flow(train_X, train_y, batch_size=32),
              epochs=30,
              validation_data=(test_X, test_y))
