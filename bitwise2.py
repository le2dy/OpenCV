import cv2
import numpy as np

src1 = np.zeros((200, 400), dtype=np.uint8)
src2 = np.zeros((200, 400), dtype=np.uint8)

src1[:, :200] = 255
src2[100:200, :] = 255

_and = cv2.bitwise_and(src1, src2)
_or = cv2.bitwise_or(src1, src2)
_xor = cv2.bitwise_xor(src1, src2)
_not = cv2.bitwise_not(src1)
_nor = cv2.bitwise_not(cv2.bitwise_or(src1, src2))

dst = np.concatenate((_and, _or, _xor, _not, _nor), axis=1)
dst = cv2.resize(dst, (1920, 1080))

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
