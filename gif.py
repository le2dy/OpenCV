import cv2
import numpy as np

cap = cv2.VideoCapture('Images/img.gif')

while cv2.waitKey(33) != 27:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, frame = cap.read()
    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()
