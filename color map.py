import cv2
import numpy as np

userColor_8UC1 = np.linspace(0, 255, num=256, endpoint=True, retstep=False, dtype=np.uint8).reshape(256, 1)
userColor_8UC3 = np.linspace(0, 255, num=256 * 3, endpoint=True, retstep=False, dtype=np.uint8).reshape(256, 1, 3)

src = cv2.imread('Images/road.png')
dst = cv2.applyColorMap(src, cv2.COLORMAP_OCEAN)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
