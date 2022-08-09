import cv2
import numpy as np

src = cv2.imread('../Images/Document/img_2.png')
src = cv2.resize(src, (500, 500))

data = src.reshape(-1, 3).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, .001)
_, label, centers = cv2.kmeans(data, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

center = centers.astype(np.uint8)
temp = center[label].reshape(src.shape)

# src = cv2.pyrMeanShiftFiltering(src, 10, 10, None, 1)
gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
dst = src.copy()

# k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# gray = cv2.erode(gray, k)
# gray = cv2.dilate(gray, k)
# gray = cv2.GaussianBlur(gray, (11, 11), 0)

_, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

# th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 5)
# th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 5)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('th', thresh)
cv2.waitKey()

z = list(contours)
i = len(z) - 1
for c in contours[::-1]:
    (x, y), radius = cv2.minEnclosingCircle(c)
    if int(radius) < 100:
        del z[i]
    i -= 1
contours = tuple(z)

print(len(contours))

# contour = contours[0]
z = list(contours)
i = len(z) - 1
for contour in contours[::-1]:
    epsilon = .05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) != 4:
        del z[i]
    i -= 1
contours = tuple(z)

contour = contours[0]
epsilon = .05 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)

temp = np.zeros((4, 2), dtype=np.float32)

idx = 0
for q in approx:
    approx[idx] = q[0]
    temp[idx] = q[0]
    idx += 1

# cv2.drawContours(dst, [approx], -1, (0, 0, 255), 3, cv2.LINE_AA)
cv2.drawContours(dst, contours, -1, (0, 0, 255), 3, cv2.LINE_AA)

sm = temp.sum(axis=1)
diff = np.diff(temp, axis=1)

print(sm)

topLeft = temp[np.argmin(sm)]
bottomRight = temp[np.argmax(sm)]
topRight = temp[np.argmin(diff)]
bottomLeft = temp[np.argmax(diff)]

z = np.where(sm == 0)
z = np.stack(z, axis=1)

pts = np.float32([topLeft, topRight, bottomRight, bottomLeft])

w1 = abs(bottomRight[0] - bottomLeft[0])
w2 = abs(topRight[0] - topLeft[0])
h1 = abs(topRight[1] - bottomRight[1])
h2 = abs(topLeft[1] - bottomLeft[1])
width = max([w1, w2])
height = max([h1, h2])

pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

matrix = cv2.getPerspectiveTransform(pts, pts2)

result = cv2.warpPerspective(src, matrix, (int(width), int(height)))
result = cv2.resize(result, None, fx=2, fy=2)

cv2.imshow('dst', dst)
# cv2.imshow('gray', gray)
# cv2.imshow('thresh', thresh)
# cv2.imshow('src', src)
# cv2.imshow('result', result)
cv2.waitKey()
cv2.destroyAllWindows()
