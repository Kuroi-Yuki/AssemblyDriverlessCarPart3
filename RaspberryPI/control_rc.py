
import io
import struct
import time
import serial
import socket

from time import sleep

HOST = '10.3.7.41'    # The remote host
PORT = 8222              # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

# Port the raspberry pi is connected to on the Arduino
serial_port = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

# Send commands to the arduino, which sends the commands to the motor
def control(command):
    print("Control received: %s" % command)
    if command == 'ff':
        serial_port.write(chr(1))
    elif command == 'bb':
        serial_port.write(chr(2))
    elif command == 'ss':
        serial_port.write(chr(0))
    elif command == 'll':
        serial_port.write(chr(3))
    elif command == 'rr':
        serial_port.write(chr(4))
    elif command == 'no':
        print("receive nothing")

def main():
    try:
        while True:
            # Receive commands from the server
            instruct = s.recv(2)
            control(instruct)
            if instruct == '':
                break

    finally:
	s.close()
        print('Connection lost')

if __name__ == "__main__":
    main()

