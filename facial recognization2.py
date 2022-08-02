import cv2

cascade_file = 'haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_file)

img = cv2.imread('Images/downtown.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_list = cascade.detectMultiScale(gray, minSize=(1, 1))

for (x, y, w, h) in face_list:
    color = (0, 0, 255)
    roi = img[y:y + h, x:x + w]
    img[y:y + h, x:x + w] = cv2.blur(roi, (10, 10))
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness = 3)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 1080, 960)
cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
