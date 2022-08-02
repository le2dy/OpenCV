import cv2


def onChange(pos):
    pass


src = cv2.imread("Images/city.png", cv2.IMREAD_GRAYSCALE)

cv2.namedWindow("Trackbar", flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow(winname='Trackbar', width=1000, height=500)

cv2.createTrackbar("threshold", "Trackbar", 0, 255, onChange)
cv2.createTrackbar("maxValue", "Trackbar", 0, 255, lambda x: x)

cv2.setTrackbarPos("threshold", "Trackbar", 127)
cv2.setTrackbarPos("maxValue", "Trackbar", 255)

while cv2.waitKey(1) != ord("q"):
    thresh = cv2.getTrackbarPos("threshold", "Trackbar")
    maxval = cv2.getTrackbarPos("maxValue", "Trackbar")

    _, binary = cv2.threshold(src, thresh, maxval, cv2.THRESH_BINARY)

    cv2.imshow("Trackbar", binary)

cv2.destroyAllWindows()
