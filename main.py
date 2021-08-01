import cv2
import numpy as np

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_eyes = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, image = cap.read()
# image = cv2.imread('elon.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 3)
        roi_color = image[y:y + h, x:x + w]
        roi_gray = gray[y:y + h, x:x + w]
        eyes = cascade_eyes.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

        cv2.imshow("image", image)

    if cv2.waitKey(1) == ord("s"):
        break

cap.release()
