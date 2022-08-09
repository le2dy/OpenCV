import cv2
import numpy as np

src = cv2.imread("../Images/skull.png")
src2 = src.copy()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

contour, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour2, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print('Figure count:', len(contour2), hierarchy)

cv2.drawContours(src, contour, -1, (0, 255, 0), 3)

for idx, count in enumerate(contour2):
    color = [int(i) for i in np.random.randint(0, 255, 3)]
    cv2.drawContours(src2, contour2, idx, color, 3)
    cv2.putText(src2, str(idx), tuple(count[0][0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    # print(idx)

cv2.imshow('src', src)
cv2.imshow('src2', src2)
cv2.waitKey(0)
cv2.destroyAllWindows()
