import cv2

src = cv2.imread('../Images/skull.png')
src = cv2.resize(src, (200, 200))

rows, cols = src.shape[:2]

m45 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.5)
m90 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1.5)

src45 = cv2.warpAffine(src, m45, (cols, rows))
src90 = cv2.warpAffine(src, m90, (cols, rows))

cv2.imshow('origin', src)
cv2.imshow('45', src45)
cv2.imshow('90', src90)

cv2.waitKey()
cv2.destroyAllWindows()
