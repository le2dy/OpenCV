import cv2
import numpy as np

src = cv2.imread('../Images/downtown.png')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 80, .01, 10)
corners = np.int32(corners)

for corner in corners:
    x, y = corner[0]
    cv2.circle(src, (x, y), 5, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()
