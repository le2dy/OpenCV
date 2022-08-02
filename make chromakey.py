import cv2

src = cv2.imread('Images/downtown.png')

x, y, w, h = cv2.selectROI(src)

src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)

crop = src_ycrcb[y: y + h, x:x + w]

channels = [1,2]
cr_bins = 128
cb_bins = 128
histSize = [cr_bins, cb_bins]
cr_range = [0, 256]
cb_range = [0, 256]
ranges = cr_range + cb_range

hist = cv2.calcHist([crop], channels, None, histSize, ranges)

hist_norm = cv2.normalize(cv2.log(hist + 1), None, 0, 255, cv2.NORM_MINMAX)
backproj =  cv2.calcBackProject([src_ycrcb], channels, hist, ranges, 1)

dst = cv2.copyTo(src, backproj)

cv2.imshow('back', backproj)
cv2.imshow('hist', hist_norm)
cv2.imshow('dst', dst)


cv2.waitKey(0)
cv2.destroyAllWindows()
