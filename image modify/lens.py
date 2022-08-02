import cv2
import numpy as np

src = cv2.imread('../Images/apple.png')
rows, cols = src.shape[:2]

exp = 5
scale = 1

mapy, mapx = np.indices((rows, cols), dtype=np.float32)

mapx = 2*mapx / (cols - 1) - 1
mapy = 2*mapy / (rows - 1) - 1

r, theta = cv2.cartToPolar(mapx, mapy)
r[r < scale] = r[r < scale] **exp

mapx, mapy = cv2.polarToCart(r, theta)

mapx = ((mapx + 1) * cols -1) / 2
mapy = ((mapy + 1) * rows -1) / 2

distorted = cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR)

cv2.imshow('src', src)
cv2.imshow('distorted', distorted)
cv2.waitKey(0)
cv2.destroyAllWindows()
