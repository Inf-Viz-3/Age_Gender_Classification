import cv2 as cv
import math
import time
import argparse
import csv
import pandas as pd
from collections.abc import Iterable

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

def scan(url):
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"

    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"

    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    # Load network
    ageNet = cv.dnn.readNet(ageModel, ageProto)
    genderNet = cv.dnn.readNet(genderModel, genderProto)
    faceNet = cv.dnn.readNet(faceModel, faceProto)

    # Open a video file or an image file or a camera stream
    cap = cv.VideoCapture(url)

    padding = 20
    while cv.waitKey(1) < 0:
        # Read frame
        hasFrame, frame = cap.read()
        if frame is None or hasFrame is None:
            return -1, -1

        if not hasFrame:
            cv.waitKey()
            break
        
        frameFace, bboxes = getFaceBox(faceNet, frame)
        if not bboxes:
            print(f"\t No face Detected, Checking next frame")
            return -1, -1

        gender_ = []
        age_ = []
        for bbox in bboxes:
            #print(f'\t Face Detected')
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            gender_.append(gender)
            
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            age_.append(age)

        return gender_, age_


def main():
    data = pd.read_excel('met.xlsx')
    df = pd.DataFrame(data, columns= ['id', 'image_url'])
    data = []

    for index, row in df.iloc[12885:].iterrows():
        print(index)
        try:
            gender, age = scan(row['image_url'])
            data.append([row['id'], age, gender])
        except:
            data.append([row['id'], -1, -1])

    df = pd.DataFrame(data, columns=['ID', 'AGE', 'GENDER'])
    df.to_csv("output2.csv", index=False)

if __name__ == "__main__":
    main()

