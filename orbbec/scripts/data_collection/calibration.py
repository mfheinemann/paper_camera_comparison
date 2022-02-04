from tkinter import OFF
from turtle import color
import numpy as np
import cv2
# from crop_target.crop_target import CropTarget
import math

from openni import openni2
from openni import _openni2 as c_api
import open3d

def main():
    print("Check that target is aligned in the frame... Press 'q' to exit")

    DEPTH_RES = [1280, 800]  # desired depth resolution
    DEPTH_RATE = 30         # desired depth frame rate
    COLOR_RES = [1280, 720]  # desired rgb resolution
    COLOR_RATE = 30         # desired rgb frame rate
    SHAPE   = 'rectangle'   # 'rectangle' 'circle'
    OFFSET = -0.02  # camera specific offset from ground truth

    # Initialize the depth device
    openni2.initialize()
    dev = openni2.Device.open_any()

    # Start the depth stream
    depth_stream = dev.create_depth_stream()
    depth_stream.start()
    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX = DEPTH_RES[0], resolutionY = DEPTH_RES[1], fps = DEPTH_RATE))

    infrared_stream = dev.create_ir_stream()
    infrared_stream.start()
    infrared_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX = 1280, resolutionY = 800, fps = DEPTH_RATE))

    # Start the color stream
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, COLOR_RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, COLOR_RES[1])
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Define target
    # if SHAPE == 'rectangle':
    #     center  = np.array([[0.0], [0.0], [1.0 + OFFSET]])    # Center of plane
    #     size    = np.array([0.5, 0.5])               # (width, height) in m
    #     angle   = np.deg2rad(0)                     # In degrees
    # elif SHAPE == 'circle':
    #     center  = np.array([[0.0], [0.0], [2.0 + OFFSET]])   # Center of shpere
    #     size    = 0.139/2.0                               # Radius in m
    #     angle   = 0.0
    # else:
    #     print("Not a valid shape!")

    # target = CropTarget()

    key = ''

    # depth_fov_h = np.deg2rad(71.5)
    # depth_fov_v = np.deg2rad(56.7)
    # color_fov_h = np.deg2rad(67.9)
    # color_fov_v = np.deg2rad(45.3)

    # fx_depth = abs((DEPTH_RES[0]/2) / math.tan(depth_fov_h/2))
    # fy_depth = abs((DEPTH_RES[1]/2) / math.tan(depth_fov_v/2))
    # print(fx_depth)
    # print(fy_depth)

    # fx_color = abs((COLOR_RES[0]/2) / math.tan(color_fov_h/2))
    # fy_color = abs((COLOR_RES[1]/2) / math.tan(color_fov_v/2))
    # print(fx_color)
    # print(fy_color)

    # intrinsic_params_depth = np.array([[980, 0, DEPTH_RES[0]/2],
    #         [0, 980, DEPTH_RES[1]/2],
    #         [0, 0, 1]])
    # intrinsic_params_color = np.array([[920, 0, COLOR_RES[0]/2],
    #         [0, 920, COLOR_RES[1]/2],
    #         [0, 0, 1]])

    # # R = np.array(extr_depth.rotation)
    # R = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float)
    # R = R.reshape(3,3)
    # # t = np.array(extr_depth.translation)
    # t = np.array([0, 0, 0], dtype=np.float)
    # t = t.reshape(3,1)
    # extrinsic_params_depth = np.concatenate((R, t), axis=1)

    # # R = np.array(extr_depth.rotation)
    # R = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float)
    # R = R.reshape(3,3)
    # # t = np.array(extr_depth.translation)
    # t = np.array([0.01, 0.01, 0], dtype=np.float)
    # t = t.reshape(3,1)    
    # extrinsic_params_color = np.concatenate((R, t), axis=1)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)*39

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    while key != 113:
        # Grab a new depth frame
        depth_frame = depth_stream.read_frame()
        depth_frame_data = depth_frame.get_buffer_as_uint16()
        # Put the depth frame into a numpy array and reshape it
        depth_image = np.frombuffer(depth_frame_data, dtype=np.int16)

        depth_image.shape = (1, 800, 1280)

        depth_image = np.swapaxes(depth_image, 0, 2)

        depth_image = np.swapaxes(depth_image, 0, 1)
        
        ret, color_frame = cap.read()


        color_image = np.asanyarray(color_frame)

        color_image = cv2.flip(color_image, 2)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)


        infrared_frame = infrared_stream.read_frame()
        infrared_frame_data = infrared_frame.get_buffer_as_triplet()
        # Put the depth frame into a numpy array and reshape it
        infrared_image = np.frombuffer(infrared_frame_data, dtype=np.int16)

        infrared_image.shape = (400, 1280, 3)

        # infrared_image = np.swapaxes(infrared_image, 0, 2)

        # infrared_image = np.swapaxes(infrared_image, 0, 1)
    
        # depth_image_with_target = target.show_target_in_image(depth_colormap, extrinsic_params_depth, intrinsic_params_depth,
        #                                 SHAPE, center, size, angle)
        # color_image_with_target = target.show_target_in_image(color_image, extrinsic_params_color, intrinsic_params_color,
        #                                 SHAPE, center, size, angle)
        # cv2.imshow("Orbbec | depth image", depth_image_with_target)
        # cv2.imshow("Orbbec | color image", color_image_with_target)

        img = color_image

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('img', img)

        ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
        #print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            ret2, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            print(mtx)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (8,6), corners2, ret)
            cv2.imshow('img', img)

        key = cv2.waitKey(1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
