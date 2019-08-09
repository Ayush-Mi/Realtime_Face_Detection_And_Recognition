# Realtime_Face_Recognition_And_Detection
This is realtime face detection and recognition using S3FD (Single Shot Scale Invariant Face Detection) model. 
## Prerequisites
Windows/Mac/Linux/RaspberryPi

Python3.3+

Pytorch

Face_recognition

Dlib

OpenCV

Flask

## Installation

Download pre-trained Dlib facial landmark prdictor model from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Download S3FD model from https://www.adrianbulat.com/downloads/python-fan/s3fd_convert.pth

## Running examples
1.Create a folder with images of known person.

2.Copy the path of the folder and paste it in live.py and client.py

3.Edit the path of .dat model in live.py, client.py and server.py

## Single System Model

Run CMD from the same folder

Run python live.py

## Client-Server Model

Run CMD from same folder

Run python server.py

Run python client.py

To run the Client-server version on different machines make sure to edit the IP address of Flask app and the port number in both the codes i.e client.py and server.py.

