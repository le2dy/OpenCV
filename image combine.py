import cv2
import numpy as np

alpha_width_rate = 15

img_face = cv2.imread("Images/face2.png")
img_skull = cv2.imread("Images/skull2.png")

print(img_face.shape, img_skull.shape)
img_face = cv2.resize(img_face, (1170, 1566))

img_comp = np.zeros_like(img_face)

height, width = img_face.shape[:2]
middle = width // 2
alpha_width = width * alpha_width_rate // 100
start = middle - alpha_width // 2
step = 100 / alpha_width

img_comp[:, :middle, :] = img_face[:, :middle, :].copy()
img_comp[:, middle:, :] = img_skull[:, middle:, :].copy()
cv2.imshow('half', img_comp)

for i in range(alpha_width + 1):
    alpha = (100 - step * i) / 100
    beta = 1 - alpha
    img_comp[:, start+i] = img_face[:, start+ i] * alpha + img_skull[:, start + i] * beta
    print(i, alpha, beta)

cv2.imshow('half2', img_comp)
cv2.waitKey()
cv2.destroyAllWindows()
