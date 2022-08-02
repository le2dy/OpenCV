import cv2
import numpy as np

rate = 5
title = 'mosaic'

src = cv2.imread('Images/skull2.png')
while True:
    x, y, w, h = cv2.selectROI(title, src, True)
    if w and h:
        roi = src[y:y + h, x:x + w]
        roi = cv2.resize(roi, (w // rate, h // rate))
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
        src[y:y + h, x:x + w] = roi
        cv2.imshow(title, src)
    else:
        break

cv2.destroyAllWindows()
