import cv2
import tensorflow as tf

new_model = tf.keras.models.load_model('cnn.h5')

cap = cv2.VideoCapture(0)
face_pattern = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceList = face_pattern.detectMultiScale(gray, 1.5)
        for (x, y, w, h) in faceList:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.rectangle(frame, (x, y-80), (x+w, y+40), (0, 0, 255), 3)
        score = new_model.predict(faceList)
        cv2.putText(im, score, (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()
