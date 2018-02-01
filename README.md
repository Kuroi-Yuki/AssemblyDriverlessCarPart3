This repository walks you through building a driverless car. It is composed of two components:

* A raspberry PI powered car
* A laptop for the ML training

## Requirements 

Set up python 3 environment on the Raspberry Pi and on the laptop. And assemble the car as per the instructions below. Install the requirements on the laptop by running `pip install -r requirements.txt`.

## Architecture

![Architecture](images/architecture.jpg)

As seen in the architecture image, the car collects sensor data and images, then they are all sent to a local server for processing. For training, the images are sent to train a model built with Keras, however when deployed, the images are passed on to this model which returns an instruction. The instruction is then passed back to the raspberry pi which in turn sends commands to the motors.

## Car

In the [Raspberry Pi Folder]('RaspberrPI/'), there are 2 files: 

* `control_rc.py`: receives commands from the computer to control the car
* `send_data_inti.py`: sends data from the car back to the computer for processing.

## Computer

In the [Computer Folder]('Computer/'), there are 4 files, two scripts and two model files:

* `collect_training_data.py`: Receives the data sent from the car and saves it on the computer.
* `model.py`: used to train the models based on the data collected from the car. The model is built from the image data collected. 
* `move_model.h5` and `steer_model.h5`: pretrained models to control the car based based on lane tracking.

## The Model

![Model](images/model.jpg)

The trained Machine Learning model is built with Keras on top of Tensorflow. It was developed by [Nvidia](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) for their self driving cars, and was adapted for the purpose of this workshop. 

![CNN](images/cnn.jpg)

In principle, the model takes in the images from the camera and then produces a command dependant on the position of the lanes in the camera frame. 
