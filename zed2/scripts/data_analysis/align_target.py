import pyzed.sl as sl
import datetime
import numpy as np
import cv2
from crop_target.crop_target import CropTarget
import math as m

def main():
    print("Check that target is aligned in the frame... Press 'q' to exit")

    depth_fps   = 15

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
        center  = np.array([[0.1], [0.05], [1.0]])    # Center of plane
        size    = np.array([0.5, 0.5])               # (width, height) in m
        angle   = 0.0                                # In radiants
    elif shape == 'circle':
        center  = np.array([[1.0], [0.0], [3.0]])   # Center of shpere
        size    = 0.2                               # Radius in m
        angle   = 0.0
    else:
        print("Not a valid shape!")

    target = CropTarget()
    cam_params = zed.get_camera_information().calibration_parameters
    pose = sl.Pose()
    depth_image = sl.Mat()
    depth_map = sl.Mat()
    b =[]
    key = ''

    while key != 113:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(depth_image, sl.VIEW.DEPTH)
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

            intrinsic_params = np.array([[cam_params.left_cam.fx, 0, cam_params.left_cam.cx],
              [0, cam_params.left_cam.fy, cam_params.left_cam.cy],
              [0, 0, 1]])

            R = pose.get_rotation_matrix(sl.Rotation()).r.T     
            t = pose.get_translation(sl.Translation()).get()            
            extrinsic_params = np.concatenate((R, np.array([t]).T), axis=1)
        
            image = depth_image.get_data()
            image_with_target = target.show_target_in_image(image, extrinsic_params, intrinsic_params,
                                            shape, center, size, angle)
            
            map = depth_map.get_data()
            a = map.shape
            b.append(map[int(a[0]/2), int(a[1]/2)])
            print(np.mean(b)/1000)
            cv2.imshow("ZED | image", image_with_target)
            key = cv2.waitKey(1)


    cv2.destroyAllWindows()
    zed.close()


if __name__ == "__main__":
    main()
