import cv2
import numpy as np
import glob

with np.load('B.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(np.uint8))

    pts1 = tuple(imgpts[0].ravel().astype(np.uint8))
    pts2 = tuple(imgpts[1].ravel().astype(np.uint8))
    pts3 = tuple(imgpts[2].ravel().astype(np.uint8))

    img = cv2.line(img, corner, pts1, (255, 0, 0), 5)
    img = cv2.line(img, corner, pts2, (0, 255, 0), 5)
    img = cv2.line(img, corner, pts3, (0, 0, 255), 5)
    return img


def draw2(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
axis2 = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0], [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

for name in glob.glob('chess/*.png'):
    img = cv2.imread(name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        _, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)

        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        imgpts2, _ = cv2.projectPoints(axis2, rvecs, tvecs, mtx, dist)

        img = draw(img, corners2, imgpts)
        img2 = draw2(img, corners2, imgpts2)
        cv2.imshow('img', img)
        cv2.imshow('img2', img2)

        key = cv2.waitKey() & 0xFF

        if key == ord('s'):
            cv2.imwrite('chess/pose.png', img)

cv2.destroyAllWindows()
