import cv2
import numpy as np

a = np.array([[1, 2, 3]], dtype=np.uint8)
b = np.array([[10, 20, 30]], dtype=np.uint8)

mask = np.array([[1, 0, 1]], dtype=np.uint8)

c1 = cv2.add(a, b, None, mask)
print(c1)
c2 = cv2.add(a, b, b.copy(), mask)
print(c2, b)
c3 = cv2.add(a, b, b, mask)
print(c3, b)
