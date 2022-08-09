import cv2
import numpy as np

src = cv2.imread("../Images/downtown.png")

edge = cv2.Laplacian(src, -1)

dst = np.hstack((src, edge))
dst = cv2.resize(dst, (1920, 1080))

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
