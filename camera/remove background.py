import cv2
import numpy as np

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('../Images/img.gif')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)

    if cv2.waitKey(delay) == 27:
        break

cap.release()
cv2.destroyAllWindows()
