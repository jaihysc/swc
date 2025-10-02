# Determines camera calibration values
# To use, put images into this folder and run this script

import numpy as np
import cv2 as cv
import glob

CHESS_GRID_X = 7
CHESS_GRID_Y = 10
CHESS_GRID = (7, 10)

# Prepare object points, (0,0,0), (1,0,0), (2,0,0) ... (6,5,0)
objp = np.zeros((CHESS_GRID_X * CHESS_GRID_Y,3), np.float32)
objp[:,:2] = np.mgrid[0:CHESS_GRID_X, 0:CHESS_GRID_Y].T.reshape(-1,2) # Set X, Y to grid

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane
for fname in glob.glob('camera/*.png'):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    # If found, add object points, image points (after refining them)
    ret, corners = cv.findChessboardCorners(gray, CHESS_GRID, None)
    if ret == True:
        objpoints.append(objp)

        # Termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # cv.drawChessboardCorners(img, CHESS_GRID, corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(0)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print('Results -----')
print(f'Reprojection error {ret}')
print('Camera matrix')
print(mtx)
print(np.linalg.inv(mtx))
print('Camera distortion')
print(dist)

cv.destroyAllWindows()