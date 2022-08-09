import cv2
import numpy as np

src = cv2.imread('../Images/skull2.png')

k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
tophat = cv2.morphologyEx(src, cv2.MORPH_TOPHAT, k)
blackhat = cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, k)

combined_np = tophat + blackhat
combined_cv2 = cv2.add(tophat, blackhat)

dst = np.hstack((tophat, blackhat, combined_np, combined_cv2))
gray = cv2.cvtColor(combined_np, cv2.COLOR_BGR2GRAY)

dst = cv2.resize(dst, (1920, 1080))

sub = combined_np - combined_cv2

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
