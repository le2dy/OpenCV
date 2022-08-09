import cv2
import numpy as np

is_picture = False

cap = cv2.VideoCapture(0)
HEIGHT, WIDTH = 540, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
src = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
vs = ()

cv2.namedWindow('src')
cv2.putText(src, 'Press any key to start camera', (70, 240), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
cv2.imshow('src', src)
cv2.waitKey()


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


def clahe(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    yuv = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return yuv


def onClick(event, x, y, flag, params):
    if event == cv2.EVENT_LBUTTONDOWN and 540 <= y < 540 + 80:
        print(x, y)
        if x in range(0, 80):
            dst = cv2.resize(params, (720, 540))
        elif x in range(80, 160):
            dst = cv2.cvtColor(params, cv2.COLOR_BGR2GRAY)
            dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        elif x in range(160, 240):
            dst = k_mean(params)
        elif x in range(240, 320):
            dst = mean_shift(params)
        elif x in range(320, 400):
            dst = clahe(params)
        elif x in range(400, 720):
            if x in range(400, 480):
                dst = thresh1(params)
            elif x in range(480, 560):
                dst = thresh2(params)
            elif x in range(560, 640):
                dst = thresh3(params)
            else:
                dst = thresh4(params)
            dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

        vs = np.vstack((dst, hs))
        cv2.imshow('src', vs)


while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    key = cv2.waitKey(delay)

    if key == 27:
        break
    elif key == ord(' '):
        cv2.imwrite('capture.jpg', frame)
        dst = frame
        break

    cv2.imshow('src', frame)

kmean = k_mean(dst)
meanshift = mean_shift(dst)
th1 = thresh1(dst)
th2 = thresh2(dst)
th3 = thresh3(dst)
th4 = thresh4(dst)
gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
yuv = clahe(dst)

kmean = cv2.resize(kmean, (80, 80))
meanshift = cv2.resize(meanshift, (80, 80))
th1 = cv2.resize(th1, (80, 80))
th2 = cv2.resize(th2, (80, 80))
th3 = cv2.resize(th3, (80, 80))
th4 = cv2.resize(th4, (80, 80))
gray = cv2.resize(gray, (80, 80))
yuv = cv2.resize(yuv, (80, 80))

th1 = cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR)
th2 = cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR)
th3 = cv2.cvtColor(th3, cv2.COLOR_GRAY2BGR)
th4 = cv2.cvtColor(th4, cv2.COLOR_GRAY2BGR)
gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

dst = cv2.resize(dst, (720, 540))
original = cv2.resize(dst, (80, 80))

hs = np.hstack((original, gray, kmean, meanshift, yuv, th1, th2, th3, th4))
vs = np.vstack((dst, hs))

cv2.imshow('src', vs)
cv2.setMouseCallback('src', onClick, dst)
cv2.waitKey()

cap.release()
cv2.destroyAllWindows()
