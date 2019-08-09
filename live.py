from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
import math
import face_recognition
import dlib

import os,sys,cv2,random,datetime,time,math
import argparse
import numpy as np

import net_s3fd
from bbox import *
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # specify path to facial landmarks .dat file

def detect(net,img):
    img = img - np.array([104,117,123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,)+img.shape)

    img = Variable(torch.from_numpy(img).float(),volatile=True)#.cuda() #uncomment to use GPU version
    BB,CC,HH,WW = img.size()
    olist = net(img)

    bboxlist = []
    for i in range(len(olist)//2): olist[i*2] = F.softmax(olist[i*2])
    for i in range(len(olist)//2):
        ocls,oreg = olist[i*2].data.cpu(),olist[i*2+1].data.cpu()
        FB,FC,FH,FW = ocls.size() 
        stride = 2**(i+2)    
        anchor = stride*4
        for Findex in range(FH*FW):
            windex,hindex = Findex%FW,Findex//FW
            axc,ayc = stride/2+windex*stride,stride/2+hindex*stride
            score = ocls[0,1,hindex,windex]
            loc = oreg[0,:,hindex,windex].contiguous().view(1,4)
            if score<0.05: continue
            priors = torch.Tensor([[axc/1.0,ayc/1.0,stride*4/1.0,stride*4/1.0]])
            variances = [0.1,0.2]
            box = decode(loc,priors,variances)
            x1,y1,x2,y2 = box[0]*1.0
            bboxlist.append([x1,y1,x2,y2,score])
    bboxlist = np.array(bboxlist)
    if 0==len(bboxlist): bboxlist=np.zeros((1, 5))
    return bboxlist


use_cuda = torch.cuda.is_available()

net = getattr(net_s3fd,'s3fd')()
net.load_state_dict(torch.load('s3fd_convert.pth')) # specify path to .pth file

#net.cuda() #uncomment to use GPU version
net.eval()


def match_boxes(box1, box2):
    flag1 = False
    import math
    box1x1 = box1.left()
    box1x2 = box1.right()
    box2x1 = box2.left()
    box2x2 = box2.right()
    box1y1 = box1.top()
    box1y2 = box1.bottom()
    box2y1 = box2.top()
    box2y2 = box2.bottom()
    xA = max(box1x1, box2x1)
    yA = max(box1y1, box2y1)
    xB = min(box1x2, box2x2)
    yB = min(box1y2, box2y2)
    interArea = (xB - xA + 1) * (yB - yA + 1)
    boxAArea = (box1x2 - box1x1 + 1) * (box1y2 - box1y1 + 1)
    boxBArea = (box2x2 - box2x1 + 1) * (box2y2 - box2y1 + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    if iou > 0.65:
        flag1 = True
    return flag1

trackers = {}
temp_trackers = {}
img_list = []
name_list = []
for names in os.listdir('C:/Users/manaksh/Desktop/Ayush/Face_Detection_Recognition/Face_Detection_Recognition/photos'): # path to folder of headshots
    img = face_recognition.load_image_file(os.path.join('C:/Users/manaksh/Desktop/Ayush/Face_Detection_Recognition/Face_Detection_Recognition/photos',names))
    img_enconding = face_recognition.face_encodings(img)[0]
    img_list.append(img_enconding)
    name_list.append(names.split(".")[0])
seen_faces = dict(zip(name_list, img_list))
count = 0
k = 0.25 # scale to resize frame. resizing improves detection speed
cap = cv2.VideoCapture(0)
while True :
    ret, img = cap.read()
    count = count +1

    frame = np.copy(img)

    
    if ret:
        for key in list(trackers):
            trackers[key].update(frame)
            if trackers[key].update(frame) < 6:
                del trackers[key]
        for key in list(temp_trackers):
            temp_trackers[key].update(frame)
            if temp_trackers[key].update(frame) < 6:
                del temp_trackers[key]
        if count%10 ==1: # frame skip rate. processes one in every 10 frames for detection. otherwise only tracks detected frames
            temp_trackers = {}
            try:
                img2 = cv2.resize(img, (int(img.shape[1]*k), int(img.shape[0]*k)))
                start = time.time()
                bounding_boxes = detect(net,img2)
                end = time.time()
                print(end - start)
                keep = nms(bounding_boxes,0.3)
                bounding_boxes = bounding_boxes[keep,:]
            except:
                bounding_boxes = []
            boxes1 = []
            for person in bounding_boxes:
                x1,y1,x2,y2,s = person
                if s<0.5: continue
                boxes1.append(dlib.rectangle(int((x1 - (x2 - x1)*0.25)/k), int((y1 - (y2 - y1)*0.25)/k), int((x2 + (x2 - x1)*0.25)/k), int((y2 + (y2 - y1)*0.25)/k)))
            boxes = boxes1
            print("len boxes is " + str(len(boxes)))
            for face_id, tracker in trackers.items():
                for box1 in boxes1:
                    a = match_boxes(box1, tracker.get_position())
                    if a:
                        del boxes[boxes == box1]
            for box in boxes:
                encoding = face_recognition.face_encodings(dlib.get_face_chip(frame, sp(frame, box)))
                if len(encoding) >0:
                    print("found encoding")
                    matches = face_recognition.face_distance(list(seen_faces.values()), encoding[0])
                    if matches[np.argmin(matches)] < 0.6:
                        face_id = list(seen_faces.keys())[np.argmin(matches)]
                        print(face_id)
                        if face_id not in trackers.keys():
                            trackers[face_id] = dlib.correlation_tracker()
                            trackers[face_id].start_track(frame, box)
                    else:
                        new_id = str(len(seen_faces.values())+1)
                        trackers[new_id] = dlib.correlation_tracker()
                        trackers[new_id].start_track(frame, box)
                        seen_faces[new_id] = encoding[0]
                else:
                    temp_new_id = str(len(temp_trackers.values())+1)
                    temp_trackers[temp_new_id] = dlib.correlation_tracker()
                    temp_trackers[temp_new_id].start_track(frame, box)
        if len(trackers)>0:
            for face_id, tracker in trackers.items():
                p1 = (int(tracker.get_position().left()), int(tracker.get_position().top()))
                p2 = (int(tracker.get_position().right()), int(tracker.get_position().bottom()))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, face_id, (int(tracker.get_position().left()) + 6, int(tracker.get_position().bottom()) - 6), font, 0.5, (255, 0, 0), 1)
        if len(temp_trackers)>0:
            for face_id, tracker in temp_trackers.items():
                p1 = (int(tracker.get_position().left()), int(tracker.get_position().top()))
                p2 = (int(tracker.get_position().right()), int(tracker.get_position().bottom()))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        cv2.imshow('test',frame)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'): break