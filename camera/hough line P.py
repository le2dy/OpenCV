import cv2
import numpy as np

src = cv2.imread('../Images/sudoku.png')
src2 = src.copy()
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 200)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, None, 20, 2)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(src2, (x1, y1), (x2, y2), (0, 255, 0), 1)

merge = np.hstack((src, src2))
cv2.imshow('merge', merge)
cv2.waitKey()
cv2.destroyAllWindows()
