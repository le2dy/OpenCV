import cv2
import numpy as np

src = cv2.imread('../Images/skull.png')

kernel_gx = np.array([[1, 0], [0, -1]])
kernel_gy = np.array([[0, 1], [-1, 0]])

edge_gx = cv2.filter2D(src, -1, kernel_gx)
edge_gy = cv2.filter2D(src, -1, kernel_gy)
merge = cv2.add(edge_gy, edge_gx)

dst = np.hstack((src, edge_gx, edge_gy, merge))

dst = cv2.resize(dst, (1920, 1080))
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
