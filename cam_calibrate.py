import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from tqdm import tqdm

width = 9
height = 6
def calibrateCamera(pathfilter):
    if Path('cam_cal.p').is_file():
        try:
            calibration = pickle.load( open('cam_cal.p', 'rb'))
            if (len(calibration) > 0):
                return calibration["mtx"], calibration["dist"]
            else:
                print('Empty pickle? hmmm.. repickle dat pickle!')
        except EOFError:
            print("Error reading cal data")
        

    print("Recalibrating")
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((height*width,3), np.float32)
    objp[:,:2] = np.mgrid[0:width, 0:height].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(pathfilter)
    img_size = ()
    # Step through the list and search for chessboard corners
    for idx, fname in tqdm(enumerate(images)):
        img = cv2.imread(fname)
        img_size=(img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (width, height), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(100)
        else:
            print('nothing found..')

    cv2.destroyAllWindows()
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "cam_cal.p", "wb" ))
    return mtx, dist