import cv2
import numpy as np

is_k_mean = False
is_mean_shift = False
is_thresh = False
one = True
two = False
three = False
four = False

cap = cv2.VideoCapture(0)
HEIGHT, WIDTH = 480, 640
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
src = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

cv2.putText(src, 'Press any key to start camera', (70, 240), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
cv2.imshow('src', src)
cv2.waitKey()


def check(fr):
    global is_k_mean

    if is_k_mean:
        th = k_mean(fr)
    elif is_mean_shift:
        th = mean_shift(fr)
    elif is_thresh:
        if one:
            th = thresh1(fr)
        elif two:
            th = thresh2(fr)
        elif three:
            th = thresh3(fr)
        elif four:
            th = thresh4(fr)
        else:
            th = thresh1(fr)
    else:
        th = fr

    return th


def k_mean(img):
    data = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, .001)
    _, label, centers = cv2.kmeans(data, 10, None, criteria, 2, cv2.KMEANS_RANDOM_CENTERS)

    center = centers.astype(np.uint8)
    return center[label].reshape(img.shape)


def mean_shift(img):
    return cv2.pyrMeanShiftFiltering(img, 10, 10, None, 1)


def thresh1(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return th


def thresh2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return th


def thresh3(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)


def thresh4(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 100, 200)


while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    key = cv2.waitKey(delay)

    if key == 27:
        break
    elif key == ord('c'):
        is_k_mean = True
        is_mean_shift = False
        is_thresh = False
        one = False
        two = False
        three = False
        four = False
    elif key == ord('s'):
        is_k_mean = False
        is_mean_shift = True
        is_thresh = False
        one = False
        two = False
        three = False
        four = False
    elif key == ord('t'):
        is_k_mean = False
        is_mean_shift = False
        is_thresh = True
        one = False
        two = False
        three = False
        four = False
    elif key == ord('n'):
        is_k_mean = False
        is_mean_shift = False
        is_thresh = False
        one = two = three = four = False
    elif key == ord('1'):
        one = True
        two = False
        three = False
        four = False
    elif key == ord('2'):
        one = False
        two = True
        three = False
        four = False
    elif key == ord('3'):
        one = False
        two = False
        three = True
        four = False
    elif key == ord('4'):
        one = False
        two = False
        three = False
        four = True

    dst = check(frame)

    cv2.imshow('src', dst)

cap.release()
cv2.destroyAllWindows()
