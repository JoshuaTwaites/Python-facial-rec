import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create() #the recognizer
recognizer.read("trainner.yml") #the trained data from faces-train

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f: #bring in the pickle file from faces-train
    original_labels = pickle.load(f)
    labels = {v:k for k,v in original_labels.items()} #inverts labels

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf>=80: #minimum confidence score to make identification
            print(id_)
            print(labels[id_], conf)
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)  # Writing in white (persons name)
            thickness = 3
            cv2.putText(frame, name, (x,y), font, 1, color, thickness, cv2.LINE_AA)

        color = (255, 255, 0) #box color
        thickness = 3
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, thickness)

    cv2.imshow('Video Capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  #when q key is pressed video window closes
        break

cap.release()
cv2.destroyAllWindows()


