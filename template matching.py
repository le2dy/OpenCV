import cv2

src = cv2.imread('Images/Screenshot.png', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('Images/letter_a.jpg', cv2.IMREAD_GRAYSCALE)
dst = cv2.imread('Images/Screenshot.png')

# template = cv2.resize(template, (40, 40))

result = cv2.matchTemplate(src, template, cv2.TM_SQDIFF_NORMED)

minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
x, y = minLoc
h, w = template.shape

dst = cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 0, 255), 1)

dst = cv2.resize(dst, (1920, 1080))

cv2.imshow('dst', dst)
cv2.imshow('template', template)
cv2.waitKey(0)
cv2.destroyAllWindows()
