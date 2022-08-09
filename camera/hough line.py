import cv2
import numpy as np

src = cv2.imread('../Images/sudoku.png')
src2 = src.copy()
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

h, w = src.shape[:2]

edge = cv2.Canny(gray, 100, 200)
lines = cv2.HoughLines(edge, 1, np.pi / 180, 130)

for line in lines:
    r, theta = line[0]
    tx, ty = np.cos(theta), np.sin(theta)
    x0, y0 = tx * r, ty * r
    cv2.circle(src2, (int(abs(x0)), int(abs(y0))), 3, (0, 0, 225), -1)
    x1, y1 = int(x0 + w * (-ty)), int(y0 + h * tx)
    x2, y2 = int(x0 - w * (-ty)), int(y0 - h * tx)
    cv2.line(src2, (x1, y1), (x2, y2), (0, 255, 0), 1)

dst = np.hstack((src, src2))
dst = cv2.resize(dst, (1080, 1080))

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
