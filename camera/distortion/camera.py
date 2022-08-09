import cv2
import numpy as np
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

obj_points = []
img_points = []

images = glob.glob('chess/*.png')

for name in images:
    img = cv2.imread(name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    if ret:
        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners)

        cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(1000)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

np.savez('B.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

img = cv2.imread('chess/img_3.png')
h, w = img.shape[:2]

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

dst = cv2.undistort(img, mtx, dist, None, new_camera_matrix)

x, y, w, h = roi
dst = dst[y:y + h, x:x + w]

cv2.imwrite('chess/calibrate.png', dst)

mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_camera_matrix, (w, h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

x, y, w, h = roi
dst = dst[y:y + h, x:x + w]

cv2.imwrite('chess/calibrate2.png', dst)

cv2.destroyAllWindows()
