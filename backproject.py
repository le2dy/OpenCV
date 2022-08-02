import cv2
import numpy as np
import matplotlib.pyplot as plt

name = 'backproject'
src = cv2.imread("Images/field.png")
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
draw = src.copy()


def masking(bp, name):
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(bp, -1, disc, bp)
    _, mask = cv2.threshold(bp, 1, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(src, src, mask=mask)
    cv2.imshow(name, result)


def manual(hist_roi):
    hist_img = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hist_rate = hist_roi / (hist_img + 1)
    h, s, v = cv2.split(hsv)
    bp = hist_rate[h.ravel(), s.ravel()]
    bp = np.minimum(bp, 1)
    bp = bp.reshape(hsv.shape[:2])
    cv2.normalize(bp, bp, 0, 255, cv2.NORM_MINMAX)
    bp = bp.astype(np.uint8)
    masking(bp, 'manual')


def cv(hist_roi):
    bp = cv2.calcBackProject([hsv], [0, 1], hist_roi, [0, 180, 0, 256], 1)
    masking(bp, 'cv')


(x, y, w, h) = cv2.selectROI(name, src, False)
if w > 0 and h > 0:
    roi = draw[y:y + h, x:x + w]
    cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 0, 255), 2)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist_roi = cv2.calcHist([hsv_roi, 0], [0, 1], None, [180, 256], [0, 180, 0, 256])
    manual(hist_roi)
    cv(hist_roi)

cv2.imshow(name, draw)
cv2.waitKey(0)
cv2.destroyAllWindows()
