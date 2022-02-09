#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
from crop_target.crop_target import CropTarget
import ir_handler

def main():
    print("Check that target is aligned in the frame... Press 'q' to exit")

    # Define target
    shape   = 'rectangle'
    if shape == 'rectangle':
        center  = np.array([[0.0], [0.0], [1.48]])    # Center of plane
        size    = np.array([0.5, 0.5])               # (width, height) in m
        angle   = np.radians(0.0)                      # In degrees                           # In radiants
    elif shape == 'circle':
        center  = np.array([[1.0], [0.0], [3.0]])   # Center of shpere
        size    = 0.2                               # Radius in m
        angle   = 0.0
    else:
        print("Not a valid shape!")
    edge_width = 0
    target  = CropTarget(shape, center, size, angle, edge_width)

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

    with dai.Device(pipeline) as device:
        # Create a receive queue for each stream
        depth_queue = device.getOutputQueue("depth", 4, blocking=False)
        disp_queue = device.getOutputQueue("disparity", 4, blocking=False)
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        # infrared settings
        #ir_handler.start_ir_handler(device)
        device.irWriteReg(0x1, 0x2b)
        device.irWriteReg(0x3, 0x3c)
        device.irWriteReg(0x4, 0x3c)

        while True:
            depth_frame = depth_queue.get().getCvFrame().astype(np.uint16)  # blocking call, will wait until a new data has arrived
            disp_frame = disp_queue.get().getCvFrame()  # blocking call, will wait until a new data has arrived
            disp_frame = getDisparityFrame(disp_frame)
            rbg_frame = rgb_queue.get().getCvFrame()  # blocking call, will wait until a new data has arrived


            calibData = device.readCalibration()
            color_intrinsic_matrix = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, 1280, 720))
            # color_extrinsic_matrix = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.RGB, dai.CameraBoardSocket.RGB))
            # color_extrinsic_matrix = color_extrinsic_matrix[:-1,:]
            # color_extrinsic_matrix[:,-1] = color_extrinsic_matrix[:,-1] / 100 # Translation is in cm for some reason
            color_extrinsic_matrix = np.hstack((np.identity(3), np.zeros((3,1))))
            
            depth_intrinsic_matrix = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, 1280, 720))
            depth_extrinsic_matrix = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.RGB, dai.CameraBoardSocket.RIGHT))
            depth_extrinsic_matrix = depth_extrinsic_matrix[:-1,:]
            depth_extrinsic_matrix[:,-1] = depth_extrinsic_matrix[:,-1] / 100 # Translation is in cm for some reason


            #image_mask = target.crop_to_target(rbg_frame, extrinsic_matrix, intrinsic_matrix)
            image_with_target = target.show_target_in_image(rbg_frame, color_extrinsic_matrix, color_intrinsic_matrix)

            depth_image_with_target = target.show_target_in_image(disp_frame, depth_extrinsic_matrix, depth_intrinsic_matrix)

            #image_concat = np.vstack((image_with_target, image_mask))
            print(depth_frame[360, 720])
            cv2.imshow("RGB", image_with_target)
            cv2.imshow("Depth", depth_image_with_target)
            if cv2.waitKey(1) == ord("q"):
                break


if __name__ == "__main__":
    main()
