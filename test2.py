import cv2
import numpy as np


def to_ascii(frame, cols=150, rows=55):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    src = np.full((1080, 1920), 255, dtype=np.uint8)
    h, w = frame.shape
    cell_w = w / cols
    cell_h = h / rows

    if cols > w or rows > h:
        raise ValueError("Error")

    # result = ""
    for i in range(rows):
        # result = ''
        for j in range(cols):
            gray = np.mean(
                frame[int(i * cell_h): min(int((i + 1) * cell_h), h), int(j * cell_w): min(int((j + 1) * cell_w), w)])
            cv2.putText(src, gray_to_char(gray), (10 + (j * 10), 10 + (i * 20)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
            # result += gray_to_char(gray)
        # result += '\n'
    src = cv2.resize(src, (1080, 960))
    cv2.imshow('src', src)
    # return result


def gray_to_char(gray):
    letter = "@%#*+=-:.   "
    # letter = letter[::-1]
    length = len(letter)
    return letter[min(int(gray * length / 255), length - 1)]


cam = cv2.VideoCapture(0)

if cam.isOpened():
    ret, frame = cam.read()

    while True:
        ret, frame = cam.read()
        # print(to_ascii(frame))
        frame = cv2.flip(frame, 1)
        to_ascii(frame)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
