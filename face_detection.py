import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x,y,w,h) in faces:
        print(x,y,w,h)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        img_item = 'my-image.png'
        cv2.imwrite(img_item, roi_gray)

        rec_color = (255, 0, 0) # BGR format 0-255
        stroke = 2
        end_coord_x = x + w
        end_coord_y = y + h
        cv2.rectangle(frame, (x, y), (end_coord_x, end_coord_y), rec_color, stroke)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 113:
        break

cap.release()
cv2.destroyAllWindows()