import cv2
import numpy as np

src1 = cv2.imread("Images/beer.png")
src2 = cv2.imread("Images/face.png")

src2 = cv2.resize(src2, (1920, 1280))

src1_gray = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
src2_gray = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

diff = cv2.absdiff(src1_gray, src2_gray)

_, diff = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)
diff_red = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
diff_red[:, :, 2] = 0

spot = cv2.bitwise_xor(src2, diff_red)

dst = np.concatenate((src1, src2, diff_red, spot), axis=1)

dst = cv2.resize(dst, (1920, 1080))

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
