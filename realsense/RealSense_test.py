# Michel Heinemann
# get average of depth data

import pyrealsense2 as rs
import numpy as np

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()      #x=945-640 y=265
        if not depth_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())

        # Get average over depth data
        depth_avg = np.mean(depth_image)

        # output
        print(depth_avg)
finally:

    # Stop streaming
    pipeline.stop()
