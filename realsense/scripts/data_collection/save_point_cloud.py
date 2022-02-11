# Michel Heinemann
# save realsense depth data over definied time period in .npz file

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import zipfile
from tkinter import messagebox
import sys


DURATION = 25            # measurement duration
LOG_PATH = '../../logs/log_rs'
RS_MODEL = 'd455'
NAME = '1'           # name of the files
DEPTH_RES = [1280, 720]  # desired depth resolution
DEPTH_RATE = 30         # desired depth frame rate
COLOR_RES = [1280, 720]  # desired rgb resolution
COLOR_RATE = 30         # desired rgb frame rate


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, DEPTH_RES[0], DEPTH_RES[1], rs.format.z16, DEPTH_RATE)
config.enable_stream(rs.stream.color, COLOR_RES[0], COLOR_RES[1], rs.format.bgr8, COLOR_RATE)

color_path = LOG_PATH + RS_MODEL + '_' + NAME + '_rgb.avi'
depth_path = LOG_PATH + RS_MODEL + '_' + NAME + '_depth.avi'
depth_array_path = LOG_PATH + RS_MODEL + '_' + NAME + '_pc'
colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), COLOR_RATE, (COLOR_RES[0], COLOR_RES[1]), 1)
depthwriter = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'XVID'), DEPTH_RATE, (DEPTH_RES[0], DEPTH_RES[1]), 1)

cfg = pipeline.start(config)
time.sleep(2)

try:
    if os.path.exists(depth_array_path):
        os.remove(depth_array_path)

    if os.path.exists(depth_array_path + '.npz'):
        if messagebox.askokcancel("Exit", "File already exists! Overwrite ?"):
            print("Overwriting!")
        else:
            print("Canceling!")
            sys.exit()

    color_frames = []
    depth_frames = []
    intrinsic_params = []
    extrinsic_params = []

    for i in range(DURATION * DEPTH_RATE):
        print("Frame: ", i)

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        #convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        colorwriter.write(color_image)
        depthwriter.write(depth_colormap)

        pc = rs.pointcloud()
        pc.map_to(color_frame)
        point_cloud = pc.calculate(depth_frame)
        point_cloud_list = np.asanyarray(point_cloud.get_vertices())
        pc = point_cloud_list.view(np.float32).reshape((point_cloud_list.size, 3))
        point_cloud_array = np.int16(1000*pc.reshape((DEPTH_RES[1], DEPTH_RES[0], 3)))

        with zipfile.ZipFile(depth_array_path, mode='a', compression=zipfile.ZIP_DEFLATED) as zf:
            array_name = str(i)
            depth_array = {array_name:point_cloud_array}
            tmpfilename = "{}.npy".format(array_name)
            np.save(tmpfilename, point_cloud_array)
            zf.write(tmpfilename)
            os.remove(tmpfilename)

        profile_1 = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
        intr = profile_1.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

        profile_2 = cfg.get_stream(rs.stream.color)
        extr = profile_1.get_extrinsics_to(profile_1)

        K = np.array([[intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]])
        intrinsic_params.append(K)

        R = np.array(extr.rotation)
        R = R.reshape(3,3)   
        t = np.array(extr.translation)
        t = t.reshape(3,1)             
        extrinsic_matrix = np.concatenate((R, t), axis=1)
        extrinsic_params.append(extrinsic_matrix)

        cv2.imshow('Stream', depth_colormap)

        if cv2.waitKey(1) == ord("q"):
            break



finally:

    npzfile = np.load(depth_array_path)    
    print(len(npzfile.files))
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
    pipeline.stop()