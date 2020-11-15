import cv2
import tensorflow as tf
import tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
import urllib.request
import time

def create_model():
    mobileNetModel = MobileNetV2(weights=None, include_top=False)
    model = Sequential()
    model.add(mobileNetModel)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='sigmoid', kernel_initializer='he_normal'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

new_model = create_model()
new_model.load_weights('helmet.hdf5')

cap = cv2.VideoCapture(0)
face_pattern = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceList = face_pattern.detectMultiScale(gray, 1.5)
        for (x, y, w, h) in faceList:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img2 = cv2.resize(img2, (100,100))
                img2 = img2.astype('float') /255.0
                img2 = img_to_array(img2)
                img3 = np.expand_dims(img2,0)
                score = new_model.predict(img3)
                cv2.putText(frame, str(score), (10,450), cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),2)
                if(score > 0.5):
                        print('true')
                else:
                        print('false')
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()
