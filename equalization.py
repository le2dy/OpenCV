import cv2
import numpy as np

src = cv2.imread("Images/downtown.png", cv2.IMREAD_GRAYSCALE)
rows, cols = src.shape[:2]

hist = cv2.calcHist([src], [0], None, [256], [0, 256])
cdf = hist.cumsum()
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) / (rows * cols) * 255
cdf = np.ma.filled(cdf_m, 0).astype('uint8')
print(cdf.shape)
src2 = cdf[src]
src3 = cv2.equalizeHist(src)

hist2 = cv2.calcHist([src2], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([src3], [0], None, [256], [0, 256])

dst = np.concatenate((src, src2, src3), axis=1)

dst = cv2.resize(dst, (1920, 1080))

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
