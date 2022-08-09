import cv2
import numpy as np
import time

src = cv2.imread('../Images/downtown.png')
rows, cols = src.shape[:2]

st = time.time()
mflip = np.float32([[-1, 0, cols - 1], [0, -1, rows - 1]])
fliped1 = cv2.warpAffine(src, mflip, (cols, rows))
print("matrix:", time.time() - st)

st2 = time.time()
mapy, mapx = np.indices((rows, cols), dtype=np.float32)
mapx = cols - mapy - 1
mapy = rows - mapy - 1
fliped2 = cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR)

cv2.imshow('o', src)
cv2.imshow('flip1', fliped1)
cv2.imshow('flip2', fliped2)
cv2.waitKey(0)
cv2.destroyAllWindows()
