import face_recognition
import dlib
import cv2
import sys, os
import math
import requests
import json
import numpy as np
import time
addr = 'http://0.0.0.0:5001'#Server IP Address
test_url = addr + '/api/test'
content_type = 'image/jpeg'
headers = {'content-type': content_type}
sp = dlib.shape_predictor("Path to shape_predictor_5_face_landmarks.dat") #Path to shape_predictor_5_face_landmarks.dat

def match_boxes(box1, box2):
    # box is a dlib rectangle
    flag1 = False
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
    if iou > 0.7:
        flag1 = True
    return flag1

#from here
trackers = {}
temp_trackers = {}
img_list = []
name_list = []
for names in os.listdir('PATH TO FOLDER OF HEADSHOTS'): 
    img = face_recognition.load_image_file(os.path.join('PATH TO FOLDER OF HEADSHOTS',names))
    img_enconding = face_recognition.face_encodings(img)[0]
    img_list.append(img_enconding)
    name_list.append(names.split(".")[0])
seen_faces = dict(zip(name_list, img_list))
count = 0
cap = cv2.VideoCapture(0)
k = 0.15
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
        if count%10 ==1:
            temp_trackers = {}
            try:
                img2 = cv2.resize(img, (int(img.shape[1]*k), int(img.shape[0] *k)))
                _, img_encoded = cv2.imencode('.jpg', img2)
                start = time.time()
                response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
                end = time.time()
                print(end -start)
                bounding_boxes = np.array(json.loads(response.text))
            except:
                bounding_boxes = []
            boxes1 = []
            for person in bounding_boxes:
                x1,y1,x2,y2,s = person
                if s<0.5: continue
                boxes1.append(dlib.rectangle(int(x1*0.95/k), int(y1*0.95/k), int(x2*1.05/k), int(y2*1.05/k)))
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
                    matches = face_recognition.compare_faces(list(seen_faces.values()), encoding[0], tolerance = 0.5)
                    print(sum(matches))
                    if any(match == True for match in matches):
                        face_id = list(seen_faces.keys())[matches.index(True)]
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
