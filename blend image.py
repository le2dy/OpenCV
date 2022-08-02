import cv2
import numpy as np
import matplotlib.pylab as plt

src1 = cv2.imread('Images/downtown.png')
src2 = cv2.imread('Images/road.png')

src1 = cv2.resize(src1, (1920, 1260))

dst1 = src1 + src2
dst2 = cv2.add(src1, src2)

images = {'src': src1, 'cap': src2, 'src + cap': dst1, 'cv2.add(src, cap)': dst2}

for i, (key, value) in enumerate(images.items()):
    plt.subplot(2, 2, i + 1)
    plt.imshow(value[:, :, ::-1])
    plt.title(key)
    plt.xticks([])
    plt.yticks([])

plt.show()
