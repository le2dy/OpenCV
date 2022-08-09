import cv2
import numpy as np

src = cv2.imread('../Images/skull.png')
src2 = np.zeros_like(src)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cnt, labels, _, center = cv2.connectedComponentsWithStats(thresh)

print(tuple(center))

for i in range(cnt):
    src2[labels == i] = [int(j) for j in np.random.randint(0, 255, 3)]
    c = tuple(center[i])
    cv2.circle(src2, (int(c[0]), int(c[1])), 3, (0, 0, 225), -1)

dst = np.hstack((src, src2))
dst = cv2.resize(dst, (1920, 1080))

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
