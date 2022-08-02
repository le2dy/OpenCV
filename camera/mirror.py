import cv2
import numpy as np

is_horizon = False
is_vertical = False

cap = cv2.VideoCapture(0)
WIDTH = 640
HEIGHT = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
rows, cols = HEIGHT, WIDTH
map_y, map_x = np.indices((rows, cols), dtype=np.float32)


def horizon(src):
    global cols
    global map_x
    global map_y
    temp_x = map_x.copy()
    temp_y = map_y.copy()
    temp_x[:, cols // 2:] = cols - temp_x[:, cols // 2:] - 1
    src = cv2.remap(src, temp_x, temp_y, cv2.INTER_LINEAR)
    return src


def vertical(src):
    temp_x = map_x.copy()
    temp_y = map_y.copy()
    temp_y[rows // 2:, :] = rows - temp_y[rows // 2:, :] - 1
    src = cv2.remap(src, temp_x, temp_y, cv2.INTER_LINEAR)
    return src


if cap.isOpened():

    while True:
        ret, frame = cap.read()
        frame = frame[:HEIGHT, :WIDTH]
        frame = cv2.flip(frame, 1)

        if is_horizon:
            frame = horizon(frame)
        elif is_vertical:
            frame = vertical(frame)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('h'):
            is_horizon = True
            is_vertical = False
        elif key == ord('v'):
            is_horizon = False
            is_vertical = True
        elif key == 27:
            is_horizon = False
            is_vertical = False

        cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()
