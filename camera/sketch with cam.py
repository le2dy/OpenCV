import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    if cv2.waitKey(33) == ord('q'):
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Laplacian(gray, -1, None, 5)
    ret, sketch = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    sketch = cv2.erode(sketch, kernel)
    sketch = cv2.medianBlur(sketch, 5)
    img_sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    paint = cv2.blur(frame, (10, 10))
    paint = cv2.bitwise_and(paint, paint, mask=sketch)

    merge = np.hstack((img_sketch, paint))
    cv2.imshow('merge', merge)

cap.release()
cv2.destroyAllWindows()
