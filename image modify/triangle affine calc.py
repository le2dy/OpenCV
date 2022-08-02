import cv2
import numpy as np

src = cv2.imread('../Images/skull2.png')
src2 = src.copy()
draw = src.copy()

pts1 = np.float32([[188, 14], [85, 202], [294, 216]])
pts2 = np.float32([[128, 40], [85, 202], [306, 167]])

x1, y1, w1, h1 = cv2.boundingRect(pts1)
x2, y2, w2, h2 = cv2.boundingRect(pts2)

roi1 = src[y1:y1 + h1, x1:x1 + w1]
roi2 = src2[y2:y2 + h2, x2:x2 + w2]

offset1 = np.zeros((3, 2), dtype=np.float32)
offset2 = np.zeros((3, 2), dtype=np.float32)

for i in range(3):
    offset1[i][0], offset1[i][1] = pts1[i][0] - x1, pts1[i][1] - y1
    offset2[i][0], offset2[i][1] = pts2[i][0] - x2, pts2[i][1] - y2

matrix = cv2.getAffineTransform(offset1, offset2)
warped = cv2.warpAffine(roi1, matrix, (w2, h2), None, cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101)

mask = np.zeros((h2, w2), dtype=np.uint8)
cv2.fillConvexPoly(mask, np.int32(offset2), (255))

warped_masked = cv2.bitwise_and(warped, warped, mask=mask)
roi2_masked = cv2.bitwise_and(roi2, roi2, mask=cv2.bitwise_not(mask))
roi2_masked = roi2_masked + warped_masked
src2[y2:y2 + h2, x2:x2 + w2] = roi2_masked

cv2.rectangle(draw, (x1, y1), (x1 + w1 , y1 + h1), (0, 255,0) , 1)
cv2.polylines(draw, [pts1.astype(np.int32)], True, (255, 0, 0), 1)
cv2.rectangle(src2, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)
cv2.imshow('origin', draw)
cv2.imshow('warped', src2)
cv2.waitKey(0)
cv2.destroyAllWindows()
