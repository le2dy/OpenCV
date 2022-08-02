import cv2

src = cv2.imread("Images/img.png")
dst = cv2.imread("Images/downtown.png")
cap = cv2.VideoCapture(0)

w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

src = cv2.resize(src, (w, h))
dst = cv2.resize(dst, (w, h))

hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (50, 150, 0), (70, 255, 255))
while True:

    ret, frame = cap.read()

    cv2.copyTo(frame, mask, src)
    cv2.imshow('frame', src)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
