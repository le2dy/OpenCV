import cv2

src = cv2.imread('../Images/skull.png')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

gray = cv2.resize(gray, (16, 16))
avg = gray.mean()

bin = 1 * (gray > avg)
print(bin)

dhash = []
for row in bin.tolist():
    s = ''.join([str(i) for i in row])
    dhash.append('%02d' % (int(s, 2)))
dhash = ''.join(dhash)
print(dhash)

cv2.namedWindow('skull', cv2.WINDOW_GUI_NORMAL)
cv2.imshow('skull', src)
cv2.waitKey()
cv2.destroyAllWindows()
