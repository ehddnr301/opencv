 import cv2
 cap = cv2.VideoCapture(0)
 face_pattern = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
 frames = []
 while True:
         ret, frame = cap.read()
         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         faceList = face_pattern.detectMultiScale(gray, 1.5)
         sky = faceList[0:10, 0:50]
         cv2.imwrite("name.jpg", sky)
         for (x, y, w, h) in faceList:
                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                 cv2.rectangle(frame, (x-20, y-20), (x+w+20, y+h+20), (0, 0, 255), 3)
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()
