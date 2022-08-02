import cv2
import matplotlib.pyplot as plt
import numpy as np

src = cv2.imread('Images/home.jpg', cv2.IMREAD_GRAYSCALE)

src_f = src.astype(np.float32)
src_norm = ((src_f - src_f.min()) * 255 / (src_f.max() - src_f.min()))
src_norm = src_norm.astype(np.uint8)

src_norm2 = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX)

hist = cv2.calcHist([src], [0], None, [256], [0, 255])
hist_norm = cv2.calcHist([src_norm], [0], None, [256], [0, 255])
hist_norm2 = cv2.calcHist([src_norm2], [0], None, [256], [0, 255])

hists = {'Before': hist, 'Manual': hist_norm, 'cv2': hist_norm2 }

cv2.imshow('Before', hist)
cv2.imshow('manual', hist_norm)
cv2.imshow('cv2', hist_norm2)

for i, (key, value) in enumerate(hists.items()):
    plt.subplot(1, 3, i + 1)
    plt.title(key)
    plt.plot(value)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
