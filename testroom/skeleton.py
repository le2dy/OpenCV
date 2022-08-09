import cv2
import numpy as np

cap = cv2.VideoCapture(0)


def distance_transform(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    skeleton = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    skeleton = (skeleton / (skeleton.max() - skeleton.min()) * 255).astype(np.uint8)

    return cv2.adaptiveThreshold(skeleton, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)


while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if cv2.waitKey(33) == ord('q'):
        break
    dst = distance_transform(frame)
    cv2.imshow('src', dst)

src = cv2.imread('../Images/skull.png', cv2.IMREAD_GRAYSCALE)

cap.release()
cv2.destroyAllWindows()
