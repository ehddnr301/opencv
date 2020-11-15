import cv2
import tensorflow as tf
import numpy as np
from skimage import transform
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential



def create_model():
    model = Sequential()
    model.add(Conv2D(64, (3,3), activation='relu', strides=(2,2), input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(126, (3,3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(126, (3,3), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

new_model = create_model()
new_model.load_weights('mwwwww.hdf5')

cap = cv2.VideoCapture(0)
face_pattern = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceList = face_pattern.detectMultiScale(gray, 1.5)
        for (x, y, w, h) in faceList:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                # cv2.rectangle(frame, (x, y-80), (x+w, y+40), (0, 0, 255), 3)
                if(w > 50 and h >50):
                        img = frame[y-80:y+40, x:x+w]
                        img = transform.resize(img, (100,100))
                        img = np.expand_dims(img,0)
                        score = new_model.predict(faceList)
                        if(score > 0.5):

                        cv2.putText(frame, str(score), (10,450), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()
