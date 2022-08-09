import cv2
import numpy as np
import matplotlib.pylab as plt

src = cv2.imread('Images/skull2.png')

channels = cv2.split(src)
colors = ('b', 'g', 'r')

cv2.imshow('src', src)

for (ch, color) in zip(channels, colors):
    hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)

plt.show()
cv2.waitKey()
cv2.destroyAllWindows()
