import cv2
import numpy as np
import glob
from time import sleep

src = cv2.imread('../Images/face.png')
cv2.imshow('src', src)

search = '../Images/101_ObjectCategories'


def img2hash(img):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (16, 16))
    avg = gray.mean()
    bi = 1 * (gray > avg)
    return bi


def hamming_distance(a, b):
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    distance = (a != b).sum()
    return distance

query_hash = img2hash(src)

paths = glob.glob(search + "/**/*.jpg")

for path in paths:
    img = cv2.imread(path)
    cv2.imshow('search', img)
    cv2.waitKey(5)
    _hash = img2hash(img)
    dst = hamming_distance(query_hash, _hash)
    sleep(500)

    if dst / 256 < 0.25:
        print(path, dst/256)
        cv2.imshow(path,img)

cv2.destroyWindow('search')
cv2.waitKey()
cv2.destroyAllWindows()
