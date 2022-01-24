#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import open3d as o3d
import math as m

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
pcl = np.zeros((resolution[0], resolution[1], 3))

print("Creating DepthAI device")
with dai.Device(pipeline) as device:
    # Create a receive queue for each stream
    depth_queue = device.getOutputQueue("depth", 4, blocking=False)
    disp_queue = device.getOutputQueue("disparity", 4, blocking=False)

    calibData = device.readCalibration()
    in_cal = o3d.camera.PinholeCameraIntrinsic()

    while True:
        depth_frame = depth_queue.get().getCvFrame().astype(np.uint16)  # blocking call, will wait until a new data has arrived
        disp_frame = disp_queue.get().getCvFrame()  # blocking call, will wait until a new data has arrived
        disp_frame = getDisparityFrame(disp_frame)

        angle = m.pi
        intrinsic_params = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, resolution[0], resolution[1]))
        in_cal.set_intrinsics(resolution[0], resolution[1], intrinsic_params[0,0], intrinsic_params[1,1], intrinsic_params[0,2],intrinsic_params[1,2])
        ext_cal = np.array([[m.cos(angle), -m.sin(angle), 0., 0.], 
                                         [m.sin(angle), m.cos(angle), 0., 0.], 
                                         [0., 0., 1., 0.], 
                                         [0., 0., 0., 1.]])
                                         
        depth_o3d = o3d.geometry.Image(depth_frame)
        pcl = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, in_cal, ext_cal, depth_scale=1000.0, depth_trunc=1000.0, stride=1)
        test = np.asarray(pcl.points)

        o3d.visualization.draw_geometries([pcl])
