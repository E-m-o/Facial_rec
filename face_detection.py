import cv2

face_cascade_front = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
face_cascade_profile = cv2.CascadeClassifier("cascades/data/haarcascade_profileface.xml")
face_cascade_eyes = cv2.CascadeClassifier("cascades/data/haarcascade_eye.xml")
cap = cv2.VideoCapture('test.avi')

while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_front = face_cascade_front.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    faces_profile = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    faces_eyes = face_cascade_eyes.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces_front:
        # print(x, y, w, h)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        rec_color = (255, 0, 0)  # BGR format 0-255
        stroke = 2
        end_coord_x = x + w
        end_coord_y = y + h
        cv2.rectangle(frame, (x, y), (end_coord_x, end_coord_y), rec_color, stroke)

    for (x, y, w, h) in faces_profile:
        # print(x, y, w, h)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        rec_color = (0, 255, 0)  # BGR format 0-255
        stroke = 2
        end_coord_x = x + w
        end_coord_y = y + h
        cv2.rectangle(frame, (x, y), (end_coord_x, end_coord_y), rec_color, stroke)

    for (x, y, w, h) in faces_eyes:
        print(x, y, w, h)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        rec_color = (0, 0, 255)  # BGR format 0-255
        stroke = 2
        end_coord_x = x + w
        end_coord_y = y + h
        cv2.rectangle(frame, (x, y), (end_coord_x, end_coord_y), rec_color, stroke)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 113:
        break

cap.release()
cv2.destroyAllWindows()

