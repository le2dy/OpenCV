import cv2
import numpy as np

src = cv2.imread('../Images/paper2.png')
src = cv2.resize(src, (1920, 1080))
rows, cols = src.shape[:2]
draw = src.copy()

marker = np.zeros((rows, cols), np.int32)
marker_id = 1
colors = []
is_drag = False


def onMouse(event, x, y, flag, params):
    global draw, marker, marker_id, is_drag

    if event == cv2.EVENT_LBUTTONDOWN:
        is_drag = True
        colors.append((marker_id, src[y, x]))
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drag:
            marker[y, x] = marker_id
            cv2.circle(draw, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow('watershed', draw)
    elif event == cv2.EVENT_LBUTTONUP:
        if is_drag:
            is_drag = False
            marker_id += 1
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.watershed(src, marker)
        draw[marker == -1] = (0, 255, 0)
        for mid, color in colors:
            draw[marker == mid] = color
        cv2.imshow('watershed', draw)


cv2.imshow('watershed', src)
cv2.setMouseCallback('watershed', onMouse)
cv2.waitKey()
cv2.destroyAllWindows()
