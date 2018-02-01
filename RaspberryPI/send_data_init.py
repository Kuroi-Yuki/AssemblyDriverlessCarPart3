import time
import threading
import io
import socket
import struct
import picamera
from mpu6050 import mpu6050
import RPi.GPIO as GPIO

MPU = mpu6050(0x68)
print (" waiting for the sensor to callibrate...")
GPIO.setmode(GPIO.BOARD)

GPIO.setwarnings(False)

INR = 1
USS = 0
PIR = 0
X1 = 0
Y1 = 0
Z1 = 0
TRIG = 13
ECHO = 12

# Connect a client socket to my_server:8000 (change my_server to the
# hostname of your server)
client_socket = socket.socket()
client_socket.connect(('10.3.7.41', 8005))

# Make a file-like object out of the connection
connection = client_socket.makefile('wb')

# create socket and bind host
client_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket2.connect(('10.3.7.41', 8007))
#connection2 = client_socket2.makefile('wb')

GPIO.setup(TRIG,GPIO.OUT)
GPIO.output(TRIG,0)
GPIO.setup(16, GPIO.IN)
GPIO.setup(22, GPIO.IN)
GPIO.setup(ECHO,GPIO.IN)

sleep(0.1)

# Get acc values
def MPU6050():
    global X1
    global Y1
    global Z1
        
    while True:
         accel_data = MPU.get_accel_data()

         Y1 = str(accel_data['y'])
         Z1 = str(accel_data['z'])         
         sleep(0.5)

# Get IR value         
def IR_sens():
    global INR
    
    while True: 
        IR = GPIO.input(16)
    
        if IR == 1:
            #print("0")
            INR = 0
            
        elif IR == 0:
            #print("1")
            INR = 1
        sleep(0.5)


def pir_sens():
    global PIR
    while True:
       PIR = GPIO.input(22)
       sleep(0.5)

# Get ultrasonic value                
def ultra():
    global USS
              
    while True:
            #print("tt")
              GPIO.output(TRIG, False)                 #Set TRIG as LOW
            ##print ("Waitng For Sensor To Settle")
              #sleep(2)                       #Delay of 2 seconds
              GPIO.output(TRIG, True)                  #Set TRIG as HIGH
              sleep(0.00001)                      #Delay of 0.00001 seconds
              GPIO.output(TRIG, False)                 #Set TRIG as LOW
              while GPIO.input(ECHO)==0:               #Check whether the ECHO is LOW
                pulse_start = time.time()              #Saves the last known time of LOW pulse
                #print("0")
                
              while GPIO.input(ECHO)==1:               #Check whether the ECHO is HIGH
                pulse_end = time.time()                #Saves the last known time of HIGH pulse 
                #print("1")
              pulse_duration = pulse_end - pulse_start #Get pulse duration to a variable
              distance = pulse_duration * 17150        #Multiply pulse duration by 17150 to get distance
              distance = round(distance, 2)            #Round to two distances
              #print(distance)
              USS = distance
              #print (distance)
              sleep(0.5)

# Print the values            
def printing():
    while True:
        print(USS, INR, PIR, X1, Y1, Z1)
        sleep(0.5)

# Send the video feed and the sensor values to the server
def send_sens():
    try:
        with picamera.PiCamera() as camera:
            camera.resolution = (320, 240)      # pi camera resolution
            camera.framerate = 5               # 10 frames/sec
            time.sleep(2)                       # give 2 secs for camera to initilize
            start = time.time()
            stream = io.BytesIO()
            
            # send jpeg format video stream
            for foo in camera.capture_continuous(stream, 'jpeg', use_video_port = True):
                connection.write(struct.pack('<L', stream.tell()))
                connection.flush()
                stream.seek(0)
                connection.write(stream.read())
                if time.time() - start > 600:
                    break
                stream.seek(0)
                stream.truncate()

                client_socket2.send('%04.0f' % USS)
                client_socket2.send('%04.0f' % INR)
                client_socket2.send('%04.0f' % PIR)
                
        connection.write(struct.pack('<L', 0))
    finally:   
            connection.close()

        
if __name__=='__main__':
    t1 = threading.Thread(target=MPU6050)
    t2 = threading.Thread(target=IR_sens)
    t3 = threading.Thread(target=ultra)
    #t4 = threading.Thread(target=printing)
    t5 = threading.Thread(target=pir_sens)
    t6 = threading.Thread(target=send_sens)
    t1.start()
    t2.start()
    t3.start()
    #t4.start()
    t5.start()
    t6.start()
   # 
   # mp = Process(target = MPU6050)
   # ir = Process(target = IR_sens)
   # us = Process(target = ultra)
   # pr = Process(target = printing)
    
    #pr.start()
    #mp.start()
    #ir.start()
    #us.start()
