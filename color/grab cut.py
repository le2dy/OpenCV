import cv2
import numpy as np

src = cv2.imread('../Images/downtown.png')
draw = src.copy()
mask = np.zeros(src.shape[:2], np.uint8)
rect = [0, 0, 0, 0]
mode = cv2.GC_EVAL
bg_model = np.zeros((1, 65), np.float64)
fg_model = np.zeros((1, 65), np.float64)


def onMouse(event, x, y, flag, params):
    global rect, mask, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        if flag <= 1:
            mode = cv2.GC_INIT_WITH_RECT
            rect[:2] = x, y
    elif event == cv2.EVENT_MOUSEMOVE and flag & cv2.EVENT_FLAG_LBUTTON:
        if mode == cv2.GC_INIT_WITH_RECT:
            temp = src.copy()
            cv2.rectangle(temp, (rect[0], rect[1]), (x, y), (0, 255, 0), 2)
            cv2.imshow('src', temp)
        elif flag > 1:
            mode = cv2.GC_INIT_WITH_MASK
            if flag & cv2.EVENT_FLAG_CTRLKEY:
                cv2.circle(draw, (x, y), 3, (255, 255, 255), -1)
                cv2.circle(mask, (x, y), 3, cv2.GC_FGD, -1)
            elif flag & cv2.EVENT_FLAG_SHIFTKEY:
                cv2.circle(draw, (x, y), 3, (0, 0, 0), -1)
                cv2.circle(mask, (x, y), 3, cv2.GC_BGD, -1)
            cv2.imshow('src', draw)
    elif event == cv2.EVENT_LBUTTONUP:
        if mode == cv2.GC_INIT_WITH_RECT:
            rect[2:] = x, y
            cv2.rectangle(draw, (rect[0], rect[1]), (x, y), (255, 0, 0), 2)
            cv2.imshow('src', draw)
        cv2.grabCut(src, mask, tuple(rect), bg_model, fg_model, 1, mode)
        src2 = src.copy()
        src2[(mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD)] = 0
        cv2.imshow('grab', src2)
        mode = cv2.GC_EVAL


cv2.imshow('src', src)
cv2.setMouseCallback('src', onMouse)
while True:
    if cv2.waitKey(0) == 27:
        break
cv2.destroyAllWindows()
