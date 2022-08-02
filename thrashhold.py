import cv2
import numpy as np
import matplotlib.pylab as plt

src = cv2.imread("Images/cup.png", cv2.IMREAD_GRAYSCALE)

thresh_np = np.zeros_like(src)
thresh_np[src > 127] = 255

ret, thresh = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY)

dst = {"Original": src, "NumPy API": thresh_np, "cv2.threshold": thresh}

for i, (key, value) in enumerate(dst.items()):
    plt.subplot(1, 3, i + 1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([])
    plt.yticks([])

cv2.imshow('thresh', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
