import cv2
import numpy as np

src = cv2.imread('../Images/skull.png', cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread('../Images/downtown.png')
h, w = src2.shape[:2]
src = cv2.resize(src, (w, h))
src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

dst = src2 - src

chroma = np.full((h, w, 3), (0, 255, 0), dtype=np.uint8)

bgr = dst[:, :, ::-1]
mask = cv2.inRange(bgr, (0, 00, 00), (255, 255, 255))

dst[mask == 0] = (0, 255, 0)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
