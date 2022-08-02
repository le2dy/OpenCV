import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    frame = cv2.flip(frame, 1)
    src = frame.copy()
    roi = src[10: 210, 200: 400]
    src[10: 210, 200: 400] = cv2.blur(roi, (100, 100))
    cv2.rectangle(src, (200, 10), (400, 210), (0, 255, 0))

    dst = np.concatenate((frame, src), axis=1)


    cv2.imshow('dst', dst)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
