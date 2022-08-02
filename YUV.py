import cv2
import numpy as np

dark = np.array([[[0, 0, 0]]], dtype=np.uint8)
middle = np.array([[[127, 127, 127]]], dtype=np.uint8)
bright = np.array([[[255, 255, 255]]], dtype=np.uint8)

d_yuv = cv2.cvtColor(dark, cv2.COLOR_BGR2YUV)
m_yuv = cv2.cvtColor(middle, cv2.COLOR_BGR2YUV)
b_yuv = cv2.cvtColor(bright, cv2.COLOR_BGR2YUV)

dst = np.concatenate((d_yuv,m_yuv,b_yuv), axis= 1)

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
