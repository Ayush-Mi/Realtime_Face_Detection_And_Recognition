# Realtime_Face_Detection_And_Recognition

This is realtime face detection and recognition using S3FD (Single Shot Scale Invariant Face Detection) model. It can be run on single system (using live.py) or dual system / cloud instances (using client.py and server.py) depending upon the resources.

## Prerequisites

Windows/Mac/Linux/RaspberryPi

Python3.3+

Pytorch

Face_recognition

Dlib

OpenCV

Flask

## Model

[shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

[s3fd_convert.7z](https://github.com/clcarwin/SFD_pytorch/releases/tag/v0.1)

## Running examples

1. Create a folder with images of known person.

2. Copy the path of the folder and paste it in live.py and client.py

3. Edit the path of .dat model in live.py, client.py and server.py

### Single System Model

Run ```python live.py``` 

### Client-Server Model

Run ```python server.py```

Wait for the server to be up and ready.

Run ```python client.py```

To run the Client-server version on different machines make sure to edit the IP address of Flask app and the port number in both the codes i.e client.py and server.py.

## References

[SFD](https://github.com/sfzhang15/SFD)

[SFD_pytorch](https://github.com/clcarwin/SFD_pytorch)
