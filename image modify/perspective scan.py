import cv2
import numpy as np

name = 'scan'
src = cv2.imread("../Images/paper2.png")

src = cv2.resize(src, (1080, 960))

rows, cols = src.shape[:2]

draw = src.copy()
pts_cnt = 0
pts = np.zeros((4, 2), dtype=np.float32)


def onMouse(event, x, y, flags, param):
    global pts_cnt
    global src

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(draw, (x, y), 10, (0, 255, 0), -1)
        cv2.imshow(name, draw)

        if pts_cnt < 4:
            pts[pts_cnt] = [x, y]
            pts_cnt += 1

        if pts_cnt == 4:
            sm = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)

            top_left = pts[np.argmin(sm)]
            bottom_right = pts[np.argmax(sm)]
            top_right = pts[np.argmin(diff)]
            bottom_left = pts[np.argmax(diff)]

            pts1 = np.float32([top_left, top_right, bottom_right, bottom_left])

            w1 = abs(bottom_right[0] - bottom_left[0])
            w2 = abs(top_right[0] - top_left[0])
            h1 = abs(top_right[1] - bottom_right[1])
            h2 = abs(top_left[1] - bottom_left[1])
            width = int(max([w1, w2]))
            height = int(max([h1, h2]))

            pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            result = cv2.warpPerspective(src, matrix, (width, height))
            cv2.imshow('result', result)

cv2.imshow(name, src)
cv2.setMouseCallback(name, onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()
