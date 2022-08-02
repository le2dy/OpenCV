import cv2
import numpy as np

src = np.full((500, 500, 3), 255, dtype=np.uint8)

cv2.namedWindow('src', cv2.WINDOW_NORMAL)
x, y = 100, 100
last_key = 0

while 1:
    cv2.imshow('src', src)
    cv2.moveWindow('src', x, y)
    key = cv2.waitKey(0)

    print(key)

    if key == ord('w'):
        y -= 10
    elif key == ord('a'):
        x -= 10
    elif key == ord('s'):
        y += 10
    elif key == ord('d'):
        x += 10
    elif key == 27:
        break
    elif key == ord('c') and last_key == ord('n'):
        print('OSDO')

    last_key = key

cv2.destroyAllWindows()
