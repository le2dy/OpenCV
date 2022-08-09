import cv2
import numpy as np

src = cv2.imread('../Images/downtown.png')
rows, cols = src.shape[:2]
mask = np.zeros((rows + 2, cols + 2), np.uint8)
new_val = (0, 255, 0)
min_diff, max_diff = (1, 1, 1), (2, 2, 2)


def onMouse(event, x, y, flags, params):
    global mask, img

    if event == cv2.EVENT_LBUTTONDOWN:
        seed = (x, y)
        ret = cv2.floodFill(src, mask, seed, new_val, min_diff, max_diff)
        cv2.imshow('src', src)


cv2.imshow('src', src)
cv2.setMouseCallback('src', onMouse)
cv2.waitKey()
cv2.destroyAllWindows()
