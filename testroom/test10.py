import cv2
import numpy as np

src = np.full((500, 500, 3), 0, dtype=np.uint8)
src = cv2.imread('../Images/skull.png')
src2 = cv2.imread('../Images/skull2.png')
h, w = src2.shape[:2]
green = np.full((h, w, 3), (0, 255, 0), dtype=np.uint8)

_, thresh = cv2.threshold(src2, 50, 255, cv2.THRESH_BINARY)

combined = cv2.add(thresh, green)
combined2 = cv2.bitwise_xor(combined, src2)
# combined3 = cv2.bitwise_xor(src, green)

cv2.imshow('combine', combined2)
cv2.waitKey()
cv2.destroyAllWindows()
