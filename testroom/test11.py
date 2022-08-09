import cv2
import numpy as np

src = cv2.imread('../Images/downtown.png')
rows, cols = src.shape[:2]
mask = np.zeros((rows + 2, cols + 2), np.uint8)
new_val = (0, 255, 0)
min_diff, max_diff = (1, 1, 1), (4, 4, 4)
is_drag = False
zero = np.zeros((rows, cols), dtype=np.uint8)
ttf = np.full((rows, cols), 255, dtype=np.uint8)

def onMouse(event, x, y, flags, params):
    global mask, img, is_drag

    if event == cv2.EVENT_LBUTTONDOWN:
        is_drag = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drag:
            seed = (x, y)
            _, _, msk, _ = cv2.floodFill(src, mask, seed, new_val, min_diff, max_diff, flags=cv2.FLOODFILL_MASK_ONLY)

            msk2 = msk[1:rows + 1, 1:cols + 1].copy()

            cv2.copyTo(ttf, msk2, zero)

            _, th = cv2.threshold(zero, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                for contour in contours:
                    epsilon = .002 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    cv2.drawContours(src, [approx], -1, (0, 255, 0), -1)
            else:
                cv2.drawContours(src, contours, -1, (0, 255, 0), -1)

            cv2.imshow('src', src)
    elif event == cv2.EVENT_LBUTTONUP:
        is_drag = False


cv2.imshow('src', src)
cv2.setMouseCallback('src', onMouse)
while True:
    key = cv2.waitKey(33)

    if key == 27:
        break
    elif key == ord('s'):
        cv2.imwrite('test1.png', src)
cv2.destroyAllWindows()
