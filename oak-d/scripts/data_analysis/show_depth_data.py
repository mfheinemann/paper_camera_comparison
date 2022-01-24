#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai

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
    #disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

    return disp


print("Creating Stereo Depth pipeline")
pipeline = dai.Pipeline()

camLeft = pipeline.create(dai.node.MonoCamera)
camRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
xoutDisparity = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

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
xoutDepth.setStreamName("depth")

camLeft.out.link(stereo.left)
camRight.out.link(stereo.right)
stereo.disparity.link(xoutDisparity.input)
stereo.depth.link(xoutDepth.input)

streams = ["disparity", "depth"]

print("Creating DepthAI device")
with dai.Device(pipeline) as device:
    # Create a receive queue for each stream
    depth_queue = device.getOutputQueue("depth", 4, blocking=False)
    disp_queue = device.getOutputQueue("disparity", 4, blocking=False)

    while True:
        depth_frame = depth_queue.get().getCvFrame().astype(np.uint16)  # blocking call, will wait until a new data has arrived
        disp_frame = disp_queue.get().getCvFrame()  # blocking call, will wait until a new data has arrived
        disp_frame = getDisparityFrame(disp_frame)

        cv2.imshow("disparity", disp_frame)

        if cv2.waitKey(1) == ord("q"):
            break
