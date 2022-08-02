import cv2
import numpy as np

fg = cv2.imread('Images/downtown.png', cv2.IMREAD_UNCHANGED)
bg = cv2.imread('Images/beer.png')

_, msk = cv2.threshold(fg[:, :, 3], 1, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(msk)

fg = cv2.cvtColor(fg, cv2.COLOR_BGRA2BGR)
h, w = fg.shape[:2]
roi = bg[10: 10 + h, 10: 10 + w]

masked_fg = cv2.bitwise_and(fg, fg, mask=msk)
masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

added = masked_fg + masked_bg
bg[10: 10 + h, 10: 10 + w] = added

cv2.imshow('dst', bg)
cv2.waitKey(0)
cv2.destroyAllWindows()
