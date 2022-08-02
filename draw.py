import cv2
import numpy as np

src = np.full((500, 500, 3), 255, dtype=np.uint8)

cv2.putText(src, "Plain", (50, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
cv2.putText(src, "Simplex", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
cv2.putText(src, "Duplex", (50, 110), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
cv2.putText(src, "Triplex", (50, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0))
cv2.putText(src, "Complex", (50, 190), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
cv2.putText(src, "Script Simplex", (50, 230), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 0))
cv2.putText(src, "Script Complex", (50, 270), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 0, 0))
cv2.putText(src, "Dead", (50, 310), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 0, 0))

cv2.imshow('src', src)
cv2.waitKey(0)
cv2.destroyAllWindows()
