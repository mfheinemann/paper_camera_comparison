#!/usr/bin/env python3

import numpy as np
import depthai as dai
import cv2
import os
import sys
import tkinter as tk
from tkinter import messagebox
import open3d as o3d
import time

DURATION = 2               # measurement duration
LOG_PATH = '../../logs/log_oak-d_pro'
NAME = '15'           # name of the files
DEPTH_RES = [1280, 720]  # desired depth resolution
DEPTH_RATE = 30         # desired depth frame rate
COLOR_RES = [1280, 720]  # desired rgb resolution
COLOR_RATE = 30         # desired rgb frame rate

def main():

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

    lrcheck      = False  # Better handling for occlusions
    extended     = False  # Closer-in minimum depth, disparity range is doubled
    subpixel     = False  # Better accuracy for longer distance, fractional disparity 32-levels
    resolution   = (1280, 720)
    medianMap    = {
        "OFF": dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF,
        "3x3": dai.StereoDepthProperties.MedianFilter.KERNEL_3x3,
        "5x5": dai.StereoDepthProperties.MedianFilter.KERNEL_5x5,
        "7x7": dai.StereoDepthProperties.MedianFilter.KERNEL_7x7,
    }
    median = medianMap["7x7"]


    def getDisparityFrame(frame):
        maxDisp = stereo.initialConfig.getMaxDisparity()
        disp = (frame * (255.0 / maxDisp)).astype(np.uint8)
        disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

        return disp


    print("Creating Stereo Depth pipeline")
    pipeline = dai.Pipeline()

    camLeft        = pipeline.create(dai.node.MonoCamera)
    camRight       = pipeline.create(dai.node.MonoCamera)
    camRgb         = pipeline.create(dai.node.ColorCamera)
    stereo         = pipeline.create(dai.node.StereoDepth)
    xoutDisparity  = pipeline.create(dai.node.XLinkOut)
    xoutDepth      = pipeline.create(dai.node.XLinkOut)
    xoutRgb        = pipeline.create(dai.node.XLinkOut)

    camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    res = dai.MonoCameraProperties.SensorResolution.THE_720_P

    for monoCam in (camLeft, camRight):  # Common config
        monoCam.setResolution(res)
        monoCam.setFps(30.0)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(median)  # KERNEL_7x7 default
    stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camRgb.setPreviewSize(1280, 720)

    xoutDisparity.setStreamName("disparity")
    xoutDepth.setStreamName("depth")
    xoutRgb.setStreamName("rgb")

    camLeft.out.link(stereo.left)
    camRight.out.link(stereo.right)
    stereo.disparity.link(xoutDisparity.input)
    stereo.depth.link(xoutDepth.input)
    camRgb.preview.link(xoutRgb.input)

    extrinsic_params_array = np.zeros((num_frames, 3, 4), dtype=np.float64)
    intrinsic_params_array = np.zeros((num_frames, 3, 3), dtype=np.float64)
    frames_array = np.zeros((num_frames,resolution[1], resolution[0], 3), dtype=np.uint16)

    with dai.Device(pipeline) as device:
        # Create a receive queue for each stream
        depth_queue = device.getOutputQueue("depth", 4, blocking=False)
        disp_queue = device.getOutputQueue("disparity", 4, blocking=False)
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        # infrared settings
        device.irWriteReg(0x1, 0x2b)
        device.irWriteReg(0x3, 0x3c)
        device.irWriteReg(0x4, 0x3c)

        time.sleep(2)
    
        for i in range(num_frames):
            print("Frame: " + str(i + 1))
            depth_frame = depth_queue.get().getCvFrame().astype(np.int16)  # blocking call, will wait until a new data has arrived
            disp_frame = disp_queue.get().getCvFrame()  # blocking call, will wait until a new data has arrived
            disp_frame = getDisparityFrame(disp_frame)
            rbg_frame = rgb_queue.get().getCvFrame()  # blocking call, will wait until a new data has arrived

            colorwriter.write(rbg_frame)
            depthwriter.write(disp_frame)

            calibData = device.readCalibration()
            intrinsic_matrix = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, 1280, 720))

            depth_extrinsic_matrix = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.RGB, dai.CameraBoardSocket.RIGHT))
            depth_extrinsic_matrix = depth_extrinsic_matrix[:-1,:]
            depth_extrinsic_matrix[:,-1] = depth_extrinsic_matrix[:,-1] / 100 # Translation is in cm for some reason

            extrinsic_params_array[i,:,:] = depth_extrinsic_matrix
            intrinsic_params_array[i,:,:] = intrinsic_matrix


            point_cloud = create_point_cloud(intrinsic_matrix, depth_frame.astype(np.int16)*-1)
            frames_array[i,:,:] = point_cloud
    
            cv2.imshow("disparity", disp_frame)
            if cv2.waitKey(1) == ord("q"):
                break


    colorwriter.release()
    depthwriter.release()
    cv2.destroyAllWindows()
    np.savez_compressed(depth_array_path, data=frames_array, 
                        intrinsic_params=intrinsic_params_array, 
                        extrinsic_params=extrinsic_params_array)


def create_point_cloud(in_params, depth_image):
    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(DEPTH_RES[0], DEPTH_RES[1], in_params[0,0], in_params[1,1], in_params[0,2], in_params[1,2])

    # PC form depth image
    pcl = o3d.geometry.PointCloud()
    pcl = pcl.create_from_depth_image(o3d.geometry.Image(depth_image), intr, project_valid_depth_only = False)

    # flip the orientation, so it looks upright, not upside-down
    pcl_points = np.asanyarray(pcl.points)
    point_cloud_array = np.int16(1000*pcl_points.reshape(DEPTH_RES[1], DEPTH_RES[0], 3))

    return point_cloud_array


if __name__ == "__main__":
    main()
