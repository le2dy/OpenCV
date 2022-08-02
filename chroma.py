import cv2

src1 = cv2.imread("Images/img.png")
src2 = cv2.imread("Images/beer.png")

w, h = src2.shape[:2]

src1 = cv2.resize(src1, (h, w))
print(src2.shape, src1.shape)

hsv = cv2.cvtColor(src1, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (50, 150, 0), (70, 255, 255))

cv2.copyTo(src2, mask, src1)

cv2.imshow('src', src1)
cv2.waitKey(0)
cv2.destroyAllWindows()
