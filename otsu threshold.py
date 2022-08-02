import cv2
import numpy as np
import matplotlib.pylab as plt

src = cv2.imread("Images/beer.png", cv2.IMREAD_GRAYSCALE)

_, t_130 = cv2.threshold(src, 130, 255, cv2.THRESH_BINARY)
t, t_otsu = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

print(t)

dst = {"Original": src, "t:130": t_130, "otsu:%d" % t: t_otsu}

for i, (key, value) in enumerate(dst.items()):
    plt.subplot(1, 3, i + 1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([])
    plt.yticks([])
