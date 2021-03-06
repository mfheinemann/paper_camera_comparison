import pyzed.sl as sl
import numpy as np
import cv2
from crop_target.crop_target import CropTarget

def main():
    print("Check that target is aligned in the frame... Press 'q' to exit")

    depth_fps   = 30    

    init = sl.InitParameters(camera_resolution = sl.RESOLUTION.HD720,
                                 camera_fps = depth_fps,
                                 depth_mode = sl.DEPTH_MODE.ULTRA,
                                 coordinate_units = sl.UNIT.MILLIMETER,
                                 coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime_params = sl.RuntimeParameters(sensing_mode = sl.SENSING_MODE.STANDARD)

    # Define target
    shape   = 'rectangle'
    if shape == 'rectangle':
        center  = np.array([[-0.029], [0.0], [1.0 - 0.015]])    # Center of plane
        size    = np.array([0.35, 0.2])                         # (width, height) in m
        angle   = np.radians(0)                                 # In degrees
    elif shape == 'circle':
        center  = np.array([[0.0], [0.0], [2.0 - 0.015]])       # Center of shperec
        size    = 0.139 / 2.0                                   # Radius in m
        angle   = np.radians(0.0)
    else:
        print("Not a valid shape!")
    edge_width = 0
    target  = CropTarget(shape, center, size, angle, edge_width)

    cam_params = zed.get_camera_information().calibration_parameters
    pose = sl.Pose()
    depth_image = sl.Mat()
    rgb_image = sl.Mat()
    key = ''

    while key != 113:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(depth_image, sl.VIEW.DEPTH)
            zed.retrieve_image(rgb_image, sl.VIEW.LEFT)

            intrinsic_params = np.array([[cam_params.left_cam.fx, 0, cam_params.left_cam.cx],
              [0, cam_params.left_cam.fy, cam_params.left_cam.cy],
              [0, 0, 1]])

            R = pose.get_rotation_matrix(sl.Rotation()).r.T     
            t = pose.get_translation(sl.Translation()).get()            
            extrinsic_params = np.concatenate((R, np.array([t]).T), axis=1)
        
            image = depth_image.get_data()
            image_with_target = target.show_target_in_image(image, extrinsic_params, intrinsic_params)

            rgb = rgb_image.get_data()
            rgb_with_target = target.show_target_in_image(rgb, extrinsic_params, intrinsic_params)

            cv2.imshow("ZED | depth", image_with_target)
            cv2.imshow("ZED | image", rgb_with_target)

            key = cv2.waitKey(1)

    cv2.destroyAllWindows()
    zed.close()


if __name__ == "__main__":
    main()
