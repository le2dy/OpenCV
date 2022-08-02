import cv2
import numpy as np

k1, k2, k3 = .5, .2, .0
# k1, k2, k3 = -.3, 0, 0

src = cv2.imread("../Images/apple.png")
rows, cols = src.shape[:2]

mapy, mapx = np.indices((rows, cols), dtype=np.float32)

mapx = 2 * mapx / (cols - 1) - 1
mapy = 2 * mapy / (rows - 1) - 1
r, theta = cv2.cartToPolar(mapx, mapy)

ru = r * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6))

mapx, mapy = cv2.polarToCart(ru, theta)
mapx = ((mapx + 1) * cols - 1) / 2
mapy = ((mapy + 1) * rows - 1) / 2

distorted = cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR)

cv2.imshow('o', src)
cv2.imshow('distort', distorted)
cv2.waitKey()
cv2.destroyAllWindows()
