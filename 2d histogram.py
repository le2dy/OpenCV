import cv2
import matplotlib.pylab as plt

plt.style.use('classic')
src = cv2.imread('Images/field.png')

plt.subplot(131)
hist = cv2.calcHist([src], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = plt.imshow(hist)
plt.title('Blue and Green')
plt.colorbar(p)

plt.subplot(132)
hist = cv2.calcHist([src], [1, 2], None, [32, 32], [0, 256, 0, 256])
p = plt.imshow(hist)
plt.title('Green and Red')
plt.colorbar(p)

plt.subplot(133)
hist = cv2.calcHist([src], [2, 0], None, [32, 32], [0, 256, 0, 256])
p = plt.imshow(hist)
plt.title('Red and Blue')
plt.colorbar(p)

plt.show()
