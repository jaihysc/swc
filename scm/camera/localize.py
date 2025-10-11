import cv2
import numpy as np
import camera

# def onClick(event, x, y, flags, params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x, y)
# img = cv2.imread('test3.png')
# cv2.imshow('image', img)
# cv2.setMouseCallback('image', onClick)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

class Localize:
    '''
    Finding 3D coordinates from 2D image

    For 2D image
    Y X -->
    |
    v
    '''
    def localize(self, imgPt, deprojPoint, debugImg) -> None:
        '''
        '''
        worldPt = np.array([
            [0, 0, 0], # Top left
            [1, 0, 0], # Top right
            [0, 1, 0]  # Bottom left
        ], np.float32)
        # imgPt = np.array([
        #     [519, 615],
        #     [735, 727],
        #     [252, 734]
        # ], np.float32)
        # deprojPoint = [376, 518]

        p3pValid = self.solveP3P(worldPt, imgPt)
        if p3pValid:
            # Reproject point 0 for Z distance (initial guess)
            REPROJ_POINT_IDX = 0
            res, distance = self.projectPoint(worldPt[REPROJ_POINT_IDX])

            # Calculate reprojection error (X, Y)
            errorReprojection = np.abs(res[0] - imgPt[REPROJ_POINT_IDX][0]) ** 2 + np.abs(res[1] - imgPt[REPROJ_POINT_IDX][1]) ** 2

            worldPoint, errorDistance = self.deprojectPoint([deprojPoint[0], deprojPoint[1], 1], distance)

        # Debug output
        FONT_ID = cv2.FONT_HERSHEY_PLAIN
        FONT_COLOR = (0, 0, 255)
        FONT_SIZE = 2.8
        FONT_THICKNESS = 2

        img = debugImg
        if p3pValid:
            cv2.putText(img, f'Reprojection error {errorReprojection}', (0, 40),  FONT_ID, FONT_SIZE, FONT_COLOR, FONT_THICKNESS)
            cv2.putText(img, f'Deproject {worldPoint}',                 (0, 80),  FONT_ID, FONT_SIZE, FONT_COLOR, FONT_THICKNESS)
            cv2.putText(img, f'Error {errorDistance}',                  (0, 120), FONT_ID, FONT_SIZE, FONT_COLOR, FONT_THICKNESS)
        else:
            cv2.putText(img, f'P3P Invalid',                            (0, 40),  FONT_ID, FONT_SIZE, FONT_COLOR, FONT_THICKNESS)

        # X Y Z axis
        imgPtInt = imgPt.astype(np.int32)
        cv2.arrowedLine(img, imgPtInt[0], imgPtInt[1], (255, 0, 0), 2)
        cv2.arrowedLine(img, imgPtInt[0], imgPtInt[2], (0, 255, 0), 2)
        # res, _ = self.projectPoint([0, 0, 1])
        # cv2.arrowedLine(img, imgPtInt[0], res.astype(np.int32), (0, 0, 255), 2)

        # X Y components of deprojected point
        if p3pValid:
            res, _ = self.projectPoint([worldPoint[0], 0, 0])
            cv2.arrowedLine(img, imgPtInt[0], res.astype(np.int32), (220, 140, 40), 2) # Line for X axis
            res, _ = self.projectPoint([0, worldPoint[1], 0])
            cv2.arrowedLine(img, imgPtInt[0], res.astype(np.int32), (40, 255, 160), 2) # Line for Y axis

            # Bottom right corner
            res, _ = self.projectPoint([1, 1, 0])
            cv2.drawMarker(img, res.astype(np.int32), (255, 255, 0), cv2.MARKER_CROSS, 16, 1)
            # Deprojected point
            cv2.drawMarker(img, deprojPoint, (0, 0, 255), cv2.MARKER_CROSS, 16, 1)

        return deprojPoint

    # Functions below rely on solveP3p results
    # can only be caled if p3pValid

    def solveP3P(self, worldPt, imgPt) -> bool:
        '''
        Solves p3p using world and image points
        worldPt: Points in world coordinates
        imgPt: Same points in image coordinates
        Returns success
        '''
        # Assuming no distortion to simplify projection
        numSolution, rvecs, tvecs = cv2.solveP3P(worldPt, imgPt, camera.CAM_MAT, None, cv2.SOLVEPNP_AP3P)
        p3pValid = numSolution > 0
        if p3pValid:
            self.rmat, _ = cv2.Rodrigues(rvecs[0]) # Use first solution (least reprojection error)
            self.rmatInv = np.linalg.inv(self.rmat)
            self.tvec = tvecs[0]
        return p3pValid

    def projectPoint(self, worldPt: list) -> np.array:
        '''
        World -> image
        Returns [X, Y, distance]
        '''
        worldPt = np.array([worldPt], np.float32).T # Transpose the vector
        res = np.matmul(camera.CAM_MAT, np.matmul(self.rmat, worldPt) + self.tvec)
        distance = res[2]

        res2 = res[0:2]
        res2 /= distance
        # column -> row, return as 1D array
        return res2.T[0], distance

    def deprojectPoint(self, imgPt: list, distance: int) -> tuple[np.array, int]:
        '''
        Image -> World
        distance: Initial distance guess, this is adjusted until world Z is 0
        Returns [X, Y, Z], error
        '''
        imgPt = np.array([imgPt], np.float32).T # Transpose the vector
        for _ in range(8):
            point = imgPt * distance

            res = np.matmul(self.rmatInv, np.matmul(camera.CAM_MAT_INV, point) - self.tvec)
            errorDistance = float(res[2]) # Z should be 0, since worldPt Z is 0

            if abs(errorDistance) < 0.01:
                break
            distance -= errorDistance
        # column -> row, return as 1D array
        return res.T[0], errorDistance