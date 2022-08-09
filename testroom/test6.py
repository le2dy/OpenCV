import cv2
import numpy as np

cap = cv2.VideoCapture(0)
flag = False


def convert(img):
    src = img
    src = cv2.resize(src, (500, 500))

    data = src.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, .001)
    _, label, centers = cv2.kmeans(data, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = centers.astype(np.uint8)
    temp = center[label].reshape(src.shape)

    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    dst = src.copy()

    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        z = list(contours)
        i = len(z) - 1
        for c in contours[::-1]:
            (x, y), radius = cv2.minEnclosingCircle(c)
            if int(radius) < 100:
                del z[i]
            i -= 1
        contours = tuple(z)

        z = list(contours)
        i = len(z) - 1
        for contour in contours[::-1]:
            epsilon = .05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) != 4:
                del z[i]
            i -= 1
        contours = tuple(z)

        contour = contours[0]
        epsilon = .05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        temp = np.zeros((4, 2), dtype=np.float32)

        if len(approx) == 4:
            idx = 0
            for q in approx:
                approx[idx] = q[0]
                temp[idx] = q[0]
                idx += 1

            cv2.drawContours(dst, [approx], -1, (0, 0, 255), 3, cv2.LINE_AA)

            sm = temp.sum(axis=1)
            diff = np.diff(temp, axis=1)

            if 0 in sm:
                return

            topLeft = temp[np.argmin(sm)]
            bottomRight = temp[np.argmax(sm)]
            topRight = temp[np.argmin(diff)]
            bottomLeft = temp[np.argmax(diff)]

            pts = np.float32([topLeft, topRight, bottomRight, bottomLeft])

            w1 = abs(bottomRight[0] - bottomLeft[0])
            w2 = abs(topRight[0] - topLeft[0])
            h1 = abs(topRight[1] - bottomRight[1])
            h2 = abs(topLeft[1] - bottomLeft[1])
            width = max([w1, w2])
            height = max([h1, h2])

            pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

            matrix = cv2.getPerspectiveTransform(pts, pts2)

            result = cv2.warpPerspective(src, matrix, (int(width), int(height)))
            result = cv2.resize(result, None, fx=2, fy=2)
            cv2.imshow('result', result)

    cv2.imshow('dst', dst)
    cv2.imshow('thresh', thresh)
    # cv2.imshow('gray', gray)
    # cv2.imshow('src', src)


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    key = cv2.waitKey(33)

    if key == 27:
        break
    elif key == ord(' '):
        flag = not flag

    if flag:
        convert(frame)
    else:
        cv2.imshow('dst', frame)

cap.release()
cv2.destroyAllWindows()
