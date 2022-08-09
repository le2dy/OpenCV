import cv2
import numpy as np

src = cv2.imread('../Images/downtown.png')

small = cv2.pyrDown(src)
big = cv2.pyrUp(small)

laplacian = cv2.subtract(src, big)

restored = big + laplacian

merge = np.hstack((src, laplacian, big, restored))
merge = cv2.resize(merge, (1920, 1080))

cv2.imshow('merge', merge)
cv2.waitKey()
cv2.destroyAllWindows()
