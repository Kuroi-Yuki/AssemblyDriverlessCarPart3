import pickle
import tensorflow as tf
import glob

# Remove tensorflow CPU error
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from PIL import Image
from keras import backend as K
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
from numpy import random
import matplotlib.image as mpimg
import csv
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Activation, Convolution2D, MaxPooling2D, ELU, Dropout
from keras.models import Sequential, Model
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_data_folder', 'trainingdata/', "Training Data Folder")
#flags.DEFINE_string('sensor_data_folder', 'sensor_data/', "Sensor Data Folder")
flags.DEFINE_string('images_folder', 'images/', "Images Folder")
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")


def load_training_data(folder):
    """
    Utility function to load training file and extract features and labels.
    Arguments:
        training_file - String
    Output:
        numpy arrays of input data split into training and validation sets
    """
    K.set_image_dim_ordering('tf')

    # Create lists to hold data and labels
    X_images_steer = []
    y_labels_steer = []
    X_sensor_move = []
    y_labels_move = []

    # Read data files
    for filename in glob.iglob(folder + '*.npz'):
        file = np.load(filename)
        X_steer_file = np.load(filename)['train_set_images']
        X_move_file = np.load(filename)['train_set_sensor']
        y_file = np.load(filename)['train_labels']
        
        for row in X_steer_file:
            X_images_steer.append(row)

        for row in X_move_file:
            X_sensor_move.append(row)
        
        for row in y_file:
            y_labels_steer.append(row[0])
            y_labels_move.append(row[1])

    # Store data as numpy arrays
    X_images_steer = np.array(X_images_steer)
    X_sensor_move = np.array(X_sensor_move)
    y_labels_steer = np.array(y_labels_steer)
    y_labels_move = np.array(y_labels_move)

    # Reshape X array as images
    X_images_steer = X_images_steer.reshape(len(X_images_steer), 240, 320, 1)
    
    print('Steer Training Data:', X_images_steer.shape)
    print('Move Training Data:', X_sensor_move.shape)
    print('Steer Labels:', y_labels_steer.shape)
    print('Move Labels:', y_labels_move.shape)

    # Split data to get un-corrected validation data
    X_steer_train, X_steer_test, y_steer_train, y_steer_test = train_test_split(X_images_steer, y_labels_steer, test_size=0.2, random_state=42)

    X_move_train, X_move_test, y_move_train, y_move_test = train_test_split(X_sensor_move, y_labels_move, test_size=0.2, random_state=42)

    print('Steer Training Data Split:', X_steer_train.shape)
    print('Move Training Data Split:', X_move_train.shape)
    print('Steer Training Labels Split:', y_steer_train.shape)
    print('Move Training Labels Split:', y_move_train.shape)
    print('Steer Test Data Split:', X_steer_test.shape)
    print('Move Test Data Split:', X_move_test.shape)
    print('Steer Test Labels Split', y_steer_test.shape)
    print('Move Test Labels Split', y_move_test.shape)

    return [(X_steer_train, y_steer_train, X_steer_test, y_steer_test), (X_move_train, y_move_train, X_move_test, y_move_test)]


def adjust_image(img):
    """
    Function that takes an image as input and returns an image as output.
    The function crops and resizes images.
    """
    image = cv2.resize(img[70:140,:], (320, 240))
    image = image.reshape(-1, 240, 320, 1)

    return image

def flip_image(img, ang):
    """
    Function that takes an image and angle as input and returns flipped image and angles as output.
    """
    image = cv2.flip(img, 1)
    angle = ang * -1.0
  
    return image, angle

def data_generator(X_train, y_train):
    """
    Function that creates batches of data to save computational resources while training.
    It takes training data array, which contains a list of images, and corresponding steering angles array.
    """
    train = np.zeros((FLAGS.batch_size, 240, 320, 1), dtype = np.float32)
    controls = np.zeros((FLAGS.batch_size, ), dtype = np.float32)

    while True:
        data, ctrl = shuffle(X_train, y_train)

        for i in range(FLAGS.batch_size):
            choice = int(np.random.choice(len(data), 1))
            if len(data[choice]) != 0:
                train[i] = adjust_image(data[choice])
                controls[i] = ctrl[choice] * (1 + np.random.uniform(-0.10,0.10))
            else:
                pass

          # Flip images randomly
            # flip_coin = np.random.randint(0,1)
            # if flip_coin == 1:
            #     train[i], controls[i] = flip_image(train[i], controls[i][])

        yield train, controls

def valid_generator(X_test, y_test):
    """
    Function that creates batches of data to save computational resources while training.
    It takes validation data array, which contains a list of images, and corresponding steering angles array.
    """
    train = np.zeros((FLAGS.batch_size, 240, 320, 1), dtype = np.dtype(np.float64))
    controls = np.zeros((FLAGS.batch_size, ), dtype = np.dtype(np.float64))

    while True:
        data, ctrls = shuffle(X_test, y_test)
        for i in range(FLAGS.batch_size):
            rand = int(np.random.choice(len(data), 1))
            if len(data[rand]) != 0:
                train[i] = adjust_image(data[rand])
                controls[i] = ctrls[rand]
            else:
                pass
        #print(train.shape)
        yield train, controls

def main(_):

    with tf.device('/gpu:0'):
        # Load steering data
        X_steer_train, y_steer_train, X_steer_test, y_steer_test = load_training_data(FLAGS.training_data_folder)[0]
        # Load sensor data  
        X_sensor_train, y_sensor_train, X_sensor_test, y_sensor_test = load_training_data(FLAGS.training_data_folder)[1]

        # Define model for steering
        steer_model = Sequential()
        steer_model.add(Lambda(lambda x: x/255-0.5, input_shape=(240, 320, 1)))
        steer_model.add(Convolution2D(24, (5, 5), padding='valid', strides=(2, 2)))
        steer_model.add(Activation('relu'))
        steer_model.add(Convolution2D(36, (5, 5), strides=(2, 2)))
        steer_model.add(Activation('relu'))
        steer_model.add(Convolution2D(48, (5, 5), strides=(2, 2)))
        steer_model.add(Activation('relu'))
        steer_model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
        steer_model.add(Activation('relu'))
        steer_model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
        steer_model.add(Activation('relu'))
        steer_model.add(Dropout(0.2))
        steer_model.add(Flatten())
        steer_model.add(Dense(1164))
        steer_model.add(Activation('relu'))
        steer_model.add(Dense(100))
        steer_model.add(Activation('relu'))
        steer_model.add(Dense(50))
        steer_model.add(Activation('relu'))
        steer_model.add(Dense(10))
        steer_model.add(Activation('relu'))
        steer_model.add(Dropout(0.5))
        steer_model.add(Dense(1))

        steer_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        print(steer_model.summary())
        
        #Train model
        steer_model.fit_generator(data_generator(X_steer_train, y_steer_train), validation_steps=15, epochs=5, steps_per_epoch=20, validation_data=valid_generator(X_steer_test, y_steer_test))

        # Save model to HDF5
        steer_model.save("steer_model.h5")


        # Define model for stop/move
        sensor_model = Sequential()
        sensor_model.add(Lambda(lambda x: x/255-0.5, input_shape=(240, 320, 1)))
        sensor_model.add(Convolution2D(24, (5, 5), padding='valid', strides=(2, 2)))
        sensor_model.add(Activation('relu'))
        sensor_model.add(Convolution2D(36, (5, 5), strides=(2, 2)))
        sensor_model.add(Activation('relu'))
        sensor_model.add(Convolution2D(48, (5, 5), strides=(2, 2)))
        sensor_model.add(Activation('relu'))
        sensor_model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
        sensor_model.add(Activation('relu'))
        sensor_model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
        sensor_model.add(Activation('relu'))
        sensor_model.add(Dropout(0.2))
        sensor_model.add(Flatten())
        sensor_model.add(Dense(1164))
        sensor_model.add(Activation('relu'))
        sensor_model.add(Dense(100))
        sensor_model.add(Activation('relu'))
        sensor_model.add(Dense(50))
        sensor_model.add(Activation('relu'))
        sensor_model.add(Dense(10))
        sensor_model.add(Activation('relu'))
        sensor_model.add(Dropout(0.5))
        sensor_model.add(Dense(1))

        sensor_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        print(sensor_model.summary())
        
        #Train model
        sensor_model.fit(X_sensor_train, y_sensor_train, validation_steps=15, epochs=5, steps_per_epoch=20, validation_data=(X_sensor_test, y_sensor_test))

        # Save model to HDF5
        sensor_model.save("move_model.h5")

if __name__ == '__main__':
    tf.app.run()