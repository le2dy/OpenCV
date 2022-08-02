import cv2
import numpy as np

l = 20
amp = 15

src = cv2.imread('../Images/downtown.png')
rows, cols = src.shape[:2]

mapy, mapx = np.indices((rows, cols), dtype=np.float32)

sinx = mapx + amp * np.sin(mapy / l)
cosy = mapy + amp * np.cos(mapx / l)

img_sinx = cv2.remap(src, sinx, mapy, cv2.INTER_LINEAR)
img_cosy = cv2.remap(src, mapx, cosy, cv2.INTER_LINEAR)
img_both = cv2.remap(src, sinx, cosy, cv2.INTER_LINEAR, None, cv2.BORDER_REPLICATE)

cv2.imshow('or', src)
cv2.imshow('sin x', img_sinx)
cv2.imshow('cos y', img_cosy)
cv2.imshow('both', img_both)
cv2.waitKey(0)
cv2.destroyAllWindows()
