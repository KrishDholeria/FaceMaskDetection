import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')

with_mask = with_mask.reshape(200, 50*50*3)
without_mask = without_mask.reshape(200, 50*50*3)

x = np.r_[with_mask, without_mask]

lables = np.zeros(x.shape[0])
lables[200:] = 1.0
names = {0:'mask', 1:'no mask'}
x_train, x_test, y_train, y_test = train_test_split(x, lables, test_size=0.25)

pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)
svm = SVC()
svm.fit(x_train, y_train)
x_test = pca.fit_transform(x_test)
y_pred = svm.predict(x_test)
# print(accuracy_score(y_test, y_pred))




font = cv2.FONT_HERSHEY_COMPLEX
capture = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier('C:/Users/LENOVO/AppData/Local/Programs/Python/Python310/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
while True:
    flag, img = capture.read()
    if flag:
        face = face_classifier.detectMultiScale(img)
        for x,y,w,h in face:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50,50))
            face = face.reshape(1, -1)
            face = pca.transform(face)
            pred = svm.predict(face)
            n = names[int(pred)]
            cv2.putText(img, n, (x,y), font, 1, (0,0,256), 2)
            # print(n)
        cv2.imshow('result',img)
        if cv2.waitKey(2) == 27:
            break
        
capture.release() 
cv2.destroyAllWindows()
