import cv2
import numpy as np
from numba import jit


@jit(nopython=True)
def to_ascii(fr, image, box_height=4, box_width=8):
    height, width = fr.shape
    for i in range(0, height, box_height):
        for j in range(0, width, box_width):
            roi = fr[i:i + box_height, j:j + box_width]
            best_match = np.inf
            best_match_index = 0
            for k in range(1, image.shape[0]):
                total_sum = np.sum(np.absolute(np.subtract(roi, image[k])))
                if total_sum < best_match:
                    best_match = total_sum
                    best_match_index = k
            roi[:, :] = image[best_match_index]
    return fr


def generate_letters():
    images = []
    letters = " \\ '(),-./:;[]_`{|}~"

    for letter in letters:
        img = np.zeros((4, 8), np.uint8)
        img = cv2.putText(img, letter, (0, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        images.append(img)
    return np.stack(images)


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
images = generate_letters()
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gb = cv2.GaussianBlur(frame, (5, 5), 0)
    canny = cv2.Canny(gb, 127, 31)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ascii_art = to_ascii(canny, images)
    cv2.imshow('ASCII', ascii_art)
    # cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
