import cv2

src = cv2.imread("Images/downtown.png")

x, y, w, h = cv2.selectROI('src', src, True)

if w and h:
    roi = src[y:y + h, x:x + w]
    cv2.imshow('roi', roi)

cv2.waitKey(0)
cv2.destroyAllWindows()
