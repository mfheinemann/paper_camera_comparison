#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
from crop_target.crop_target import CropTarget

resolutionMap = {"800": (1280, 800), "720": (1280, 720), "400": (640, 400)}
resolution = resolutionMap["720"]

outRectified = False  # Output and display rectified streams
lrcheck = False  # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = False  # Better accuracy for longer distance, fractional disparity 32-levels

medianMap = {
    "OFF": dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF,
    "3x3": dai.StereoDepthProperties.MedianFilter.KERNEL_3x3,
    "5x5": dai.StereoDepthProperties.MedianFilter.KERNEL_5x5,
    "7x7": dai.StereoDepthProperties.MedianFilter.KERNEL_7x7,
}
median = medianMap["7x7"]


def getDisparityFrame(frame):
    maxDisp = stereo.initialConfig.getMaxDisparity()
    disp = (frame * (255.0 / maxDisp)).astype(np.uint8)

    return disp


print("Creating Stereo Depth pipeline")
pipeline = dai.Pipeline()

camLeft = pipeline.create(dai.node.MonoCamera)
camRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
xoutDisparity = pipeline.create(dai.node.XLinkOut)

camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
res = (
    dai.MonoCameraProperties.SensorResolution.THE_800_P
    if resolution[1] == 800
    else dai.MonoCameraProperties.SensorResolution.THE_720_P
    if resolution[1] == 720
    else dai.MonoCameraProperties.SensorResolution.THE_400_P
)
for monoCam in (camLeft, camRight):  # Common config
    monoCam.setResolution(res)
    # monoCam.setFps(20.0)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(median)  # KERNEL_7x7 default
stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
stereo.setLeftRightCheck(lrcheck)
stereo.setExtendedDisparity(extended)
stereo.setSubpixel(subpixel)

xoutDisparity.setStreamName("disparity")

camLeft.out.link(stereo.left)
camRight.out.link(stereo.right)
stereo.disparity.link(xoutDisparity.input)

streams = ["disparity", "depth"]


# Define target
shape   = 'rectangle'
if shape == 'rectangle':
    center  = np.array([[1.0], [0.0], [2.0]])    # Center of plane
    size    = np.array([0.5, 0.5])               # (width, height) in m
    angle   = 0.9                                # In radiants
elif shape == 'circle':
    center  = np.array([[1.0], [0.0], [3.0]])   # Center of shpere
    size    = 0.2                               # Radius in m
    angle   = 0.0
else:
    print("Not a valid shape!")
edge_width = 0
target  = CropTarget(shape, center, size, angle, edge_width)

print("Creating DepthAI device")
with dai.Device(pipeline) as device:
    # Create a receive queue for each stream
    disp_queue = device.getOutputQueue("disparity", 4, blocking=False)
    calibData = device.readCalibration()

    while True:
        disp_frame = disp_queue.get().getCvFrame()  # blocking call, will wait until a new data has arrived
        disp_frame = getDisparityFrame(disp_frame)

        intrinsic_params = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, resolution[0], resolution[1]))

        R = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
        t = np.zeros((3,1))         
        extrinsic_params = np.concatenate((R, t), axis=1)
    
        image_with_target = target.crop_to_target(disp_frame, extrinsic_params, intrinsic_params)
        cv2.imshow("OAK-D | image", image_with_target)

        if cv2.waitKey(1) == ord("q"):
            break
