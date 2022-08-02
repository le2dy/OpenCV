import cv2

cascade_file = 'haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_file)

img = cv2.imread('Images/test.png')


def image_detector(img, cascade):
    img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    results = cascade.detectMultiScale(gray,
                                       scaleFactor=1.5,
                                       minNeighbors=5,
                                       minSize=(20, 20))
    for box in results:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)

    cv2.imshow('face', img)
    cv2.waitKey(0)


image_detector(img, cascade)
