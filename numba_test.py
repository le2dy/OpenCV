import cv2
import numpy as np
import cProfile
from numba import jit


@jit(nopython=True)
def process(frame, box_h=6, box_w=16):
    h, w, _ = frame.shape
    for i in range(0, h, box_h):
        for j in range(0, w, box_w):
            roi = frame[i:i + box_h, j:j + box_w]
            b_mean = np.mean(roi[:, :, 0])
            g_mean = np.mean(roi[:, :, 1])
            r_mean = np.mean(roi[:, :, 2])
            roi[:, :, 0] = b_mean
            roi[:, :, 1] = g_mean
            roi[:, :, 2] = r_mean
    return frame


def main(iterations=300):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 580)

    for _ in range(iterations):
        _, frame = cap.read()

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        frame = process(frame)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


main(iterations=1)
cProfile.run("main(iterations=300)")
