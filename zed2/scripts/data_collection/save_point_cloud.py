import ogl_viewer.viewer as gl
import pyzed.sl as sl
import datetime
import numpy as np
import cv2

def main():
    show_image  = True
    depth_fps   = 30
    sim_seconds = 5

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
    cam_params = zed.get_camera_information().calibration_parameters
    pose = sl.Pose()

    depth_image = sl.Mat()
    depth_map  = sl.Mat()
    frames = []
    intrinsic_params = []
    extrinsic_params = []

    num_frames = sim_seconds * depth_fps
    for i in range(num_frames):
        print("Frame: " + str(i + 1))
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(depth_map, sl.MEASURE.MEASURE.XYZ)
            cloud = depth_cloud.get_data()

            if show_image:
                zed.retrieve_image(depth_image, sl.VIEW.DEPTH)
                image = depth_image.get_data()
                cv2.imshow("ZED | image", image)
                cv2.waitKey(1)
            
            cloud_uint16 = np.uint16(cloud)
            frames.append(cloud_uint16)

            K = np.array([[cam_params.left_cam.fx, 0, cam_params.left_cam.cx],
              [0, cam_params.left_cam.fy, cam_params.left_cam.cy],
              [0, 0, 1]])
            intrinsic_params.append(K)

            R = pose.get_rotation_matrix(sl.Rotation()).r.T     
            t = pose.get_translation(sl.Translation()).get()            
            extrinsic_matrix = np.concatenate((R, np.array([t]).T), axis=1)
            extrinsic_params.append(extrinsic_matrix)


    cv2.destroyAllWindows()
    zed.close()

    extrinsic_params_array = np.stack(extrinsic_params, axis=0)
    intrinsic_params_array = np.stack(intrinsic_params, axis=0)
    frames_array = np.stack(frames, axis=0)

    current_datetime = datetime.datetime.now()
    video_title = "../../logs/log_zed2_" + current_datetime.strftime("%y%m%d-%H%M%S")
    np.savez_compressed(video_title, data=frames_array, 
                        intrinsic_params=intrinsic_params_array, 
                        extrinsic_params=extrinsic_params_array)


if __name__ == "__main__":
    main()
