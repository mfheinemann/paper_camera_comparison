from multiprocessing.connection import wait
from tkinter import OFF
import numpy as np
import cv2
from crop_target.crop_target import CropTarget
import pyrealsense2 as rs
import time

def main():
    print("Check that target is aligned in the frame... Press 'q' to exit")

    DEPTH_RES = [1280, 720]  # desired depth resolution
    DEPTH_RATE = 30         # desired depth frame rate
    COLOR_RES = [1280, 720]  # desired rgb resolution
    COLOR_RATE = 30         # desired rgb frame rate
    SHAPE   = 'circle'   # 'rectangle' 'circle'
    # rs435
    OFFSET = -0.01  # camera specific offset from ground truth
    # rs455
    OFFSET = -0.012

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, DEPTH_RES[0], DEPTH_RES[1], rs.format.z16, DEPTH_RATE)
    config.enable_stream(rs.stream.color, COLOR_RES[0], COLOR_RES[1], rs.format.bgr8, COLOR_RATE)
    # config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    # config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

    cfg = pipeline.start(config)
    time.sleep(2.0)

    # Define target
    if SHAPE == 'rectangle':
        center  = np.array([[0.0], [0.0], [2.0 + OFFSET]])    # Center of plane
        size    = np.array([0.5, 0.5])               # (width, height) in m
        angle   = np.deg2rad(0)                     # In degrees
    elif SHAPE == 'circle':
        center  = np.array([[0.0], [0.0], [1.0 + OFFSET]])   # Center of shpere
        size    = 0.139/2.0                               # Radius in m
        angle   = 0.0
    else:
        print("Not a valid shape!")
    edge_width = 0
    target  = CropTarget(SHAPE, center, size, angle, edge_width)

    profile_depth = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
    profile_color = cfg.get_stream(rs.stream.color)
    # profile_inf1 = cfg.get_stream(rs.stream.infrared, 1)
    # profile_inf2 = cfg.get_stream(rs.stream.infrared, 2)
    intr_depth = profile_depth.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
    intr_color = profile_color.as_video_stream_profile().get_intrinsics()
    print(intr_depth)

    extr_depth = profile_depth.get_extrinsics_to(profile_depth)
    extr_color = profile_depth.get_extrinsics_to(profile_color)
    print(extr_depth)

    key = ''
    intrinsic_params_depth = np.array([[intr_depth.fx, 0, intr_depth.ppx],
            [0, intr_depth.fy, intr_depth.ppy],
            [0, 0, 1]])
    intrinsic_params_color = np.array([[intr_color.fx, 0, intr_color.ppx],
            [0, intr_color.fy, intr_color.ppy],
            [0, 0, 1]])

    R = np.array(extr_depth.rotation)
    R = R.reshape(3,3)
    t = np.array(extr_depth.translation)
    t = t.reshape(3,1)     
    extrinsic_params_depth = np.concatenate((R, t), axis=1)

    R = np.array(extr_color.rotation)
    R = R.reshape(3,3)
    R = np.swapaxes(R, 1, 0) # only requiered for D435
    t = np.array(extr_color.translation)
    t = t.reshape(3,1)     
    extrinsic_params_color = np.concatenate((R, t), axis=1)

    while key != 113:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    
        depth_image_with_target = target.show_target_in_image(depth_colormap, extrinsic_params_depth, intrinsic_params_depth)
        color_image_with_target = target.show_target_in_image(color_image, extrinsic_params_color, intrinsic_params_color)
        cv2.imshow("RealSense | depth image", depth_image_with_target)
        cv2.imshow("RealSense | color image", color_image_with_target)
        key = cv2.waitKey(1)
    cv2.destroyAllWindows()
    pipeline.stop()


if __name__ == "__main__":
    main()
