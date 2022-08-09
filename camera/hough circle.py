import cv2
import numpy as np

src = cv2.imread('../Images/candy.png')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)

circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 2, 30, None, 200)

if circles is not None:
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        cv2.circle(src, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(src, (i[0], i[1]), 2, (0, 0, 255), 5)

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()
