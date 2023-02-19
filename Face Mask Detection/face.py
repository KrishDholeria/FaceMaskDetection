import cv2
import os
import numpy as np


capture = cv2.VideoCapture(0)
data = []
face_classifier = cv2.CascadeClassifier('C:/Users/LENOVO/AppData/Local/Programs/Python/Python310/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
while True:
    flag, img = capture.read()
    if flag:
        face = face_classifier.detectMultiScale(img)
        for x,y,w,h in face:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50,50))
            print(len(data))
            if len(data) < 200:
                data.append(face)
        cv2.imshow('result',img)
        if cv2.waitKey(2) == 27 or len(data) >= 200:
            break
        
capture.release()
cv2.destroyAllWindows()

# cv2.imshow(data[0])

np.save('without_mask.npy', data)