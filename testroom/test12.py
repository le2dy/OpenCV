import cv2

chroma = cv2.imread('test2.png')
src = cv2.imread('../Images/road.png')

h, w = chroma.shape[:2]

src = cv2.resize(src, (w, h))

hsv = cv2.cvtColor(chroma, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (50, 50, 0), (80, 255, 255))

cv2.copyTo(src, mask, chroma)

cv2.imshow('chroma', chroma)
cv2.waitKey()
cv2.destroyAllWindows()
