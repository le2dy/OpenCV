import cv2
import numpy as np

target = cv2.imread('../Images/letter_A.jpg')
shapes = cv2.imread('../Images/Screenshot.png')

target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
shapes_gray = cv2.cvtColor(shapes, cv2.COLOR_BGR2GRAY)

_, target_thresh = cv2.threshold(target_gray, 127, 255, cv2.THRESH_BINARY_INV)
_, shapes_thresh = cv2.threshold(shapes_gray, 127, 255, cv2.THRESH_BINARY_INV)

target_contours, _ = cv2.findContours(target_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
shapes_contours, _ = cv2.findContours(shapes_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

matches = []

for ctr in shapes_contours:
    match = cv2.matchShapes(target_contours[0], ctr, cv2.CONTOURS_MATCH_I2, 0.0)
    matches.append((match, ctr))
    cv2.putText(shapes, '%.1f' % match, tuple(ctr[0][0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

matches.sort(key=lambda x: x[0])
cv2.drawContours(shapes, [matches[0][1]], -1, (0, 255, 0), 3)
cv2.drawContours(target, target_contours[0], -1, (0, 255, 0), 3)

shapes = cv2.resize(shapes, (1080, 640))

cv2.imshow('target', target)
cv2.imshow('match', shapes)
cv2.waitKey()
cv2.destroyAllWindows()
