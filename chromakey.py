import cv2
import numpy as np

src1 = cv2.imread("Images/chromakey.png")
src2 = cv2.imread('Images/field.png')

h1, w1 = src1.shape[:2]
h2, w2 = src2.shape[:2]

x = (w2 - w1) // 2
y = h2 - h1
w = x + w1
h = y + h1

chromakey = src1[:10, :10, :]
offset = 20

hsv_chroma = cv2.cvtColor(chromakey, cv2.COLOR_BGR2HSV)
hsv_img = cv2.cvtColor(src1, cv2.COLOR_BGR2HSV)

chroma_h = hsv_chroma[:, :, 0]
lower = np.array([chroma_h.min() - offset, 100, 100])
upper = np.array([chroma_h.max() + offset, 255, 255])

mask = cv2.inRange(hsv_img, lower, upper)
mask_inv = cv2.bitwise_not(mask)
roi = src2[y:h, x:w]
fg = cv2.bitwise_and(src1, src1, mask=mask_inv)
bg = cv2.bitwise_and(roi, roi, mask=mask)
src2[y:h, x:w] = fg + bg

cv2.imshow('chromakey', src1)
cv2.imshow('added', src2)
cv2.waitKey(0)
cv2.destroyAllWindows()
