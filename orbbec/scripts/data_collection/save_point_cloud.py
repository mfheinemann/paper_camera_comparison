# Michel Heinemann
# save realsense depth data over definied time period in .npz file

from openni import openni2
from openni import _openni2 as c_api
import open3d

import numpy as np
import cv2
import os
import time
import zipfile
from tkinter import messagebox
import sys


DURATION = 25            # measurement duration
LOG_PATH = '../../logs/log_'
RS_MODEL = 'orbbec'
NAME = '4'           # name of the files
DEPTH_RES = [1280, 800]  # desired depth resolution
DEPTH_RATE = 30         # desired depth frame rate
COLOR_RES = [1280, 720]  # desired rgb resolution
COLOR_RATE = 30         # desired rgb frame rate

# Initialize the depth device
openni2.initialize()
dev = openni2.Device.open_any()

# Start the depth stream
depth_stream = dev.create_depth_stream()
depth_stream.start()
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX = DEPTH_RES[0], resolutionY = DEPTH_RES[1], fps = DEPTH_RATE))

# Start the color stream
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, COLOR_RES[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, COLOR_RES[1])
if not cap.isOpened():
    print("Cannot open camera")
    exit()


color_path = LOG_PATH + RS_MODEL + '_' + NAME + '_rgb.avi'
depth_path = LOG_PATH + RS_MODEL + '_' + NAME + '_depth.avi'
depth_array_path = LOG_PATH + RS_MODEL + '_' + NAME + '_pc'
colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), COLOR_RATE, (COLOR_RES[0], COLOR_RES[1]), 1)
depthwriter = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'XVID'), DEPTH_RATE, (DEPTH_RES[0], DEPTH_RES[1]), 1)

# cfg = pipeline.start(config)

try:
    if os.path.exists(depth_array_path):
        os.remove(depth_array_path)

    if os.path.exists(depth_array_path + '.npz'):
        if messagebox.askokcancel("Exit", "File already exists! Overwrite ?"):
            print("Overwriting!")
        else:
            print("Canceling!")
            sys.exit()

    t_start = time.time()
    t_current = 0

    color_frames = []
    depth_frames = []
    timestamps = []
    intrinsic_params = []
    extrinsic_params = []

    i=1

    while True:

        timestamps.append(time.time())

        # Grab a new depth frame
        depth_frame = depth_stream.read_frame()
        depth_frame_data = depth_frame.get_buffer_as_uint16()
        # Put the depth frame into a numpy array and reshape it

        depth_image = np.frombuffer(depth_frame_data, dtype=np.uint16)

        depth_image.shape = (1, 800, 1280)

        depth_image = np.swapaxes(depth_image, 0, 2)

        depth_image = np.swapaxes(depth_image, 0, 1)
        
        ret, color_frame = cap.read()

        color_image = np.asanyarray(color_frame)
        color_image = cv2.flip(color_image, 1)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        colorwriter.write(color_image)
        depthwriter.write(depth_colormap)

        intr = open3d.camera.PinholeCameraIntrinsic()
        intr.set_intrinsics(1280, 800, 945.028, 945.028, 640, 400)

        # PC form depth image
        pcd = open3d.geometry.PointCloud()
        pcd = pcd.create_from_depth_image(open3d.geometry.Image(depth_image), intr, project_valid_depth_only = False)

        # flip the orientation, so it looks upright, not upside-down
        pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]])

        pc = np.asanyarray(pcd.points)

        point_cloud_array = np.int16(1000*pc.reshape((DEPTH_RES[1], DEPTH_RES[0], 3)))

        # visualize point cloud
        # pcd_visual = pcd.create_from_depth_image(open3d.geometry.Image(depth_image), intr, project_valid_depth_only = True)

        # open3d.visualization.draw_geometries([pcd_visual])

        with zipfile.ZipFile(depth_array_path, mode='a', compression=zipfile.ZIP_DEFLATED) as zf:
            array_name = str(t_current)
            depth_array = {array_name:point_cloud_array}
            tmpfilename = "{}.npy".format(array_name)
            np.save(tmpfilename, point_cloud_array)
            zf.write(tmpfilename)
            os.remove(tmpfilename)

        K = np.array([[970, 0, DEPTH_RES[0]/2],
            [0, 960, DEPTH_RES[1]/2],
            [0, 0, 1]])
        intrinsic_params.append(K)
        
        R = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float)
        R = R.reshape(3,3)
        # t = np.array(extr_depth.translation)
        t = np.array([0, 0.04, 0], dtype=np.float)
        t = t.reshape(3,1)
        extrinsic_matrix = np.concatenate((R, t), axis=1)
        extrinsic_params.append(extrinsic_matrix)

        cv2.imshow('Stream', depth_colormap)

        if cv2.waitKey(1) == ord("q"):
            break

        if int(t_current) != int(time.time() - t_start):
            print("Time recorded: {} ...".format(int(t_current)))
        t_current = time.time() - t_start

        if int(i/DEPTH_RATE) == int(DURATION):
            break

        i += 1


finally:

    npzfile = np.load(depth_array_path)    
    print(len(npzfile.files))
    #timestamps_array = np.stack(timestamps, axis=0)
    frames_array = np.zeros((len(npzfile.files), DEPTH_RES[1], DEPTH_RES[0], 3), dtype=np.int16)
    i = 0
    for key, value in npzfile.items():
        frames_array[i,:,:,:] = value
        i += 1

    extrinsic_params_array = np.stack(extrinsic_params, axis=0)
    intrinsic_params_array = np.stack(intrinsic_params, axis=0)
    np.savez_compressed(depth_array_path, data=frames_array, 
                        intrinsic_params=intrinsic_params_array, 
                        extrinsic_params=extrinsic_params_array)
    
    if os.path.exists(depth_array_path):    # deleting zip archive, .npz is save
        os.remove(depth_array_path)

    colorwriter.release()
    depthwriter.release()
    cv2.destroyAllWindows()