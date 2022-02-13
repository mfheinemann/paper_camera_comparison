import pyzed.sl as sl
import numpy as np
import cv2
import os
import sys
import tkinter as tk
from tkinter import messagebox
from matplotlib import pyplot as plt

def main():
    DURATION = 25                # measurement duration
    LOG_PATH = '../../logs/log_zed2'
    NAME = '8'           # name of the files
    DEPTH_RES = [1280, 720]  # desired depth resolution
    DEPTH_RATE = 30         # desired depth frame rate
    COLOR_RES = [1280, 720]  # desired rgb resolution
    COLOR_RATE = 30         # desired rgb frame rate
    num_frames = DURATION * DEPTH_RATE

    color_path = LOG_PATH + '_' + NAME + '_rgb.avi'
    depth_path = LOG_PATH + '_' + NAME + '_depth.avi'
    depth_array_path = LOG_PATH + '_' + NAME + '_pc'
    colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), COLOR_RATE, (COLOR_RES[0], COLOR_RES[1]), 1)
    depthwriter = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'XVID'), DEPTH_RATE, (DEPTH_RES[0], DEPTH_RES[1]), 1)

    if os.path.exists(depth_array_path + '.npz'):
        root = tk.Tk()
        root.withdraw() 
        if messagebox.askokcancel("Exit", "File already exists! Overwrite ?"):
            print("Overwriting!")
        else:
            print("Canceling!")
            sys.exit()

    init = sl.InitParameters(camera_resolution = sl.RESOLUTION.HD720,
                                 camera_fps = DEPTH_RATE,
                                 depth_mode = sl.DEPTH_MODE.ULTRA,
                                 coordinate_units = sl.UNIT.MILLIMETER,
                                 coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime_params = sl.RuntimeParameters(sensing_mode = sl.SENSING_MODE.STANDARD)
    cam_params = zed.get_camera_information().calibration_parameters
    pose = sl.Pose()

    depth_image = sl.Mat()
    depth_map  = sl.Mat()
    rgb_image = sl.Mat()
    extrinsic_params_array = np.zeros((num_frames, 3, 4), dtype=np.float64)
    intrinsic_params_array = np.zeros((num_frames, 3, 3), dtype=np.float64)
    frames_array = np.zeros((num_frames,DEPTH_RES[1], DEPTH_RES[0], 4), dtype=np.uint16)

    for i in range(num_frames):
        print("Frame: " + str(i + 1))
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(depth_map, sl.MEASURE.XYZ)
            zed.retrieve_image(depth_image, sl.VIEW.DEPTH)
            zed.retrieve_image(rgb_image, sl.VIEW.LEFT)

            cloud = depth_map.get_data()
            depth_frame = depth_image.get_data()
            color_frame = rgb_image.get_data()
            
            colorwriter.write(color_frame[:,:,:-1])
            depthwriter.write(depth_frame[:,:,:-1])
            cloud[:,:,2] = - cloud[:,:,2] 
            cloud_uint16 = cloud.astype(np.uint16)

            K = np.array([[cam_params.left_cam.fx, 0, cam_params.left_cam.cx],
              [0, cam_params.left_cam.fy, cam_params.left_cam.cy],
              [0, 0, 1]])

            R = pose.get_rotation_matrix(sl.Rotation()).r.T     
            t = pose.get_translation(sl.Translation()).get()            
            extrinsic_matrix = np.concatenate((R, np.array([t]).T), axis=1)

            extrinsic_params_array[i,:,:] = extrinsic_matrix
            intrinsic_params_array[i,:,:] = K
            frames_array[i,:,:,:] = cloud_uint16

            cv2.imshow("ZED | rgb_image", depth_frame)
            cv2.waitKey(1)


    colorwriter.release()
    depthwriter.release()
    cv2.destroyAllWindows()
    zed.close()

    np.savez_compressed(depth_array_path, data=frames_array, 
                        intrinsic_params=intrinsic_params_array, 
                        extrinsic_params=extrinsic_params_array)


if __name__ == "__main__":
    main()
