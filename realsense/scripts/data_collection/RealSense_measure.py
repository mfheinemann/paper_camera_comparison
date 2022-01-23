# Michel Heinemann
# save realsense depth data over definied time period in .npz file

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import zipfile

DURATION = 5            # measurement duration
NAME = 'test'           # name of the files
DEPTH_RES = [640, 480]  # desired depth resolution
DEPTH_RATE = 30         # desired depth frame rate
COLOR_RES = [640, 480]  # desired rgb resolution
COLOR_RATE = 30         # desired rgb frame rate


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, DEPTH_RES[0], DEPTH_RES[1], rs.format.z16, DEPTH_RATE)
config.enable_stream(rs.stream.color, COLOR_RES[0], COLOR_RES[1], rs.format.bgr8, COLOR_RATE)

color_path = NAME + '_rgb.avi'
depth_path = NAME + '_depth.avi'
depth_array_path = NAME + '_pc'
colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), COLOR_RATE, (COLOR_RES[0], COLOR_RES[1]), 1)
depthwriter = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'XVID'), DEPTH_RATE, (DEPTH_RES[0], DEPTH_RES[1]), 1)

pipeline.start(config)

try:
    if os.path.exists(depth_array_path):
        os.remove(depth_array_path)

    t_start = time.time()
    t_end = 5
    t_current = 0

    color_frames = []
    depth_frames = []
    timestamps = []

    i=1

    while True:

        timestamps.append(time.time())

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

        with zipfile.ZipFile(depth_array_path, mode='a', compression=zipfile.ZIP_DEFLATED) as zf:
            array_name = str(t_current)
            depth_array = {array_name:depth_image}
            tmpfilename = "{}.npy".format(array_name)
            np.save(tmpfilename, depth_image)
            zf.write(tmpfilename)
            os.remove(tmpfilename)

        cv2.imshow('Stream', depth_colormap)

        if cv2.waitKey(1) == ord("q"):
            break

        if int(t_current) != int(time.time() - t_start):
            print("Time recorded: {} ...".format(int(t_current)))
        t_current = time.time() - t_start

        if int(i/DEPTH_RATE) == int(DURATION):
            break

        i += 1

    npzfile = np.load(depth_array_path)    
    print(len(npzfile.files))
    timestamps_array = np.stack(timestamps, axis=0)
    frames_array = np.stack(npzfile.files, axis=0)

finally:
    np.savez_compressed(depth_array_path, data=frames_array, timestamp=timestamps_array)

    colorwriter.release()
    depthwriter.release()
    cv2.destroyAllWindows()
    pipeline.stop()