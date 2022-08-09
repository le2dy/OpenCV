import cv2
import numpy as np
import matplotlib.pylab as plt

# src = cv2.imread('Images/downtown.png')
src = cv2.imread('testroom/capture.jpg')
src_yuv = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)

src_eq = src_yuv.copy()
src_eq[:, :, 0] = cv2.equalizeHist(src_eq[:, :, 0])
src_eq = cv2.cvtColor(src_eq, cv2.COLOR_YUV2BGR)

src_clahe = src_yuv.copy()
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
src_clahe[:, :, 0] = clahe.apply(src_clahe[:, :, 0])
src_clahe = cv2.cvtColor(src_clahe, cv2.COLOR_YUV2BGR)

dst = np.concatenate((src, src_clahe, src_eq), axis=1)

dst = cv2.resize(dst, (1920, 1080))

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
