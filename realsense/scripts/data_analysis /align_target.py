import numpy as np
import cv2
from crop_target.crop_target import CropTarget
import pyrealsense2 as rs

def main():
    print("Check that target is aligned in the frame... Press 'q' to exit")

    DEPTH_RES = [640, 480]  # desired depth resolution
    DEPTH_RATE = 30         # desired depth frame rate
    COLOR_RES = [640, 480]  # desired rgb resolution
    COLOR_RATE = 30         # desired rgb frame rate
    SHAPE   = 'rectangle'   # 'rectangle' 'circle'

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, DEPTH_RES[0], DEPTH_RES[1], rs.format.z16, DEPTH_RATE)
    config.enable_stream(rs.stream.color, COLOR_RES[0], COLOR_RES[1], rs.format.bgr8, COLOR_RATE)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

    cfg = pipeline.start(config)

    # Define target
    if SHAPE == 'rectangle':
        center  = np.array([[0.0], [0.0], [1.0]])    # Center of plane
        size    = np.array([0.5, 0.5])               # (width, height) in m
        angle   = np.deg2rad(45)                     # In degrees
    elif SHAPE == 'circle':
        center  = np.array([[1.0], [0.0], [3.0]])   # Center of shpere
        size    = 0.2                               # Radius in m
        angle   = 0.0
    else:
        print("Not a valid shape!")

    target = CropTarget()
    profile_1 = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
    intr = profile_1.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
    print(intr)

    profile_2 = cfg.get_stream(rs.stream.color)
    extr = profile_2.get_extrinsics_to(profile_2)
    print(extr)

    key = ''
    intrinsic_params = np.array([[intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]])

    R = np.array(extr.rotation)
    R = R.reshape(3,3)
    t = np.array(extr.translation)
    t = t.reshape(3,1)     
    extrinsic_params = np.concatenate((R, t), axis=1)

    while key != 113:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
    
        depth_image_with_target = target.show_target_in_image(depth_image, extrinsic_params, intrinsic_params,
                                        SHAPE, center, size, angle)
        color_image_with_target = target.show_target_in_image(color_image, extrinsic_params, intrinsic_params,
                                        SHAPE, center, size, angle)
        cv2.imshow("RealSense | depth image", depth_image_with_target)
        cv2.imshow("RealSense | color image", color_image_with_target)
        key = cv2.waitKey(1)
    cv2.destroyAllWindows()
    pipeline.stop()


if __name__ == "__main__":
    main()
