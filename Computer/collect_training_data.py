
import os
import threading
import SocketServer
import socket
import cv2
import numpy as np
import math
import time
import sys

from time import sleep

import msvcrt # uses curses for mac

HOST = '10.3.5.186'      # Host address of the server

class VideoStreamHandler(SocketServer.StreamRequestHandler):

    frame = 1

    def handle(self):
        global sensor_data

        self.HOST = '10.3.8.247'       # Port through which the video feed is received
        self.PORT = 8222              # Arbitrary non-privileged port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((HOST, self.PORT))
        self.s.listen(0)
        self.conn, self.addr = self.s.accept()

        self.HOST2 = '10.3.8.247'      # Port through which the sensor data is received
        self.PORT2 = 8223             # Arbitrary non-privileged port
        self.s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s2.bind((HOST, self.PORT2))
        self.s2.listen(0)
        self.conn2, self.addr2 = self.s2.accept()

        # Sensor data initialization
        self.ultrasonic = 0
        self.ir = 0
        self.pir = 0

        stream_bytes = ' ' # Used to read images from video feed

        # To be used to  accumulate training data (image + sensor data + labels for movement + labels for steering) 
        self.image_array = np.zeros((1, 76800))
        self.label_array = np.zeros((1, 2), 'float')
        self.sensor_data_array = np.zeros((1, 3), 'int')

        # create labels representing car movement (0:backward, 1:nothing, 2:stop, 3:forward)
        n1 = 0

        # create labels representing steering (0:left, 1:send nothing, 2:right)
        n2 = 0

        # Prepare folders to store original image, preprocessed image, and training data
        if not os.path.exists("original_images"):
            os.makedirs("original_images")
        if not os.path.exists("training_images"):
            os.makedirs("training_images")  
        if not os.path.exists("training_data"):
            os.makedirs("training_data")  
       

        # stream video frames one by one
        try:
            while True:
                stream_bytes += self.rfile.read(1024)
                first = stream_bytes.find('\xff\xd8')
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last+2]
                    stream_bytes = stream_bytes[last+2:]

                    #Recieve sensor data from the Raspberry PI through web sockets
                    self.ultrasonic = int(self.conn2.recv(4))
                    self.ir = int(self.conn2.recv(4))
                    self.pir = int(self.conn2.recv(4))

                    # Get the image and save it save it to the folder original_images
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
                    cv2.imwrite('original_images/frame{:>05}.jpg'.format(self.frame), image)

                    # Change image to gray scale, save it to the folder training_images and display it
                    gray = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_GRAYSCALE)
                    cv2.imwrite('training_images/frame{:>05}.jpg'.format(self.frame), gray)
                    cv2.imshow('Grayscale', gray)

                    # Get the key pressed (use curses on mac)
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        #print(key)
                        if key == 'w':              # Forward
                            self.conn.sendall('ff')   # Send msg 'ff' to Raspberry PI to instruct the car movement
                            n1 = 2                   # Select the appropriate label in array for car movement
                        elif key == 'p':            # Stop
                            self.conn.sendall('ss') # Send msg 'ss' to Raspberry PI to instruct the car movement
                            n1 = 1                  # Select the appropriate label in array for car movement
                        elif key == 's':            # Backward
                            self.conn.sendall('bb') # Send msg 'bb' to Raspberry PI to instruct the car movement
                            n1 = -1                # Select the appropriate label in array for car movement
                        if key == 'a':              # Left
                            self.conn.sendall('ll') # Send msg 'll' to Raspberry PI to instruct steering the car
                            n2 = -1                 # Select the appropriate label in array for car steering
                        elif key == 'd':            # Right
                            self.conn.sendall('rr') # Send msg 'rr' to Raspberry PI to instruct steering the car
                            n2 = 1                  # Select the appropriate label in array for car steering
                        elif ord(key) == 27:        # Detect esc key # Press esc key to stop recieving images and capturing and sending instructions
                            self.conn.sendall('no') # Send msg 'no' to send nothing to the car
                            n2 = 0                   # Select the appropriate label in array for car steering and movement
                            n1 = 0                  
                            break               

                        else:
                            self.conn.sendall('no') # Send msg 'no' to send nothing to the car
                            n2 = 0                  # Select the appropriate label in array for car steering
                            n1 = 0
                    else:
                        self.conn.sendall('no')
                        n2 = 0
                        n1 = 0

                     # Reshape the 240x320 image matrix to an array
                    temp_arr = gray.reshape(1, 76800).astype(np.float32)

                    # Add label for the current image captured for car movement and steering to the stack of all accumulated labels
                    self.label_array = np.vstack((self.label_array, [n1, n2]))
                    
                    # Add image array (pixels) for the current image to the stack of other accumulated image arrays
                    self.image_array = np.vstack((self.image_array, temp_arr))                                

                    # Add sensor data for the current image to the stack of other accumulated sensor data
                    self.sensor_data_array = np.vstack((self.sensor_data_array, [self.ultrasonic, self.ir, self.pir]))                                 

                    self.frame += 1

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
            # save training images, sensor data and labels
            train_set_images = self.image_array[1:, :]
            train_set_sensor = self.sensor_data_array[1:, :]
            #print (train_set_sensor)
            train_labels = self.label_array[1:, :]

            # save training data as a numpy file # To be fed to train the model
            file_name = str(int(time.time()))
            try:    
                np.savez('training_data/' + file_name + '.npz', train_set_images=train_set_images, train_set_sensor=train_set_sensor, train_labels=train_labels)
            except IOError as e:
                print(e)

            # Print the description of the dimensionality of each component of the training data 
            print(train_set_images.shape)
            print(train_set_sensor.shape)
            print(train_labels.shape)

            cv2.destroyAllWindows()

        finally:
            self.s.close()
            self.s2.close()
            print ("Connection closed")
            sys.exit(0)


class ThreadServer(object):

    def server_thread(host, port):
        server = SocketServer.TCPServer((host, port), VideoStreamHandler)
        server.serve_forever()

    video_thread = threading.Thread(name='VideoStream', target=server_thread('10.3.8.247', 8220))
    video_thread.start()

if __name__ == '__main__':
    ThreadServer()

