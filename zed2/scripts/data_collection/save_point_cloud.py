import ogl_viewer.viewer as gl
import pyzed.sl as sl
import time
import datetime
import numpy as np
import pyransac3d as pyrsc

def main():
    init = sl.InitParameters(camera_resolution = sl.RESOLUTION.VGA,
                                 camera_fps = 30,
                                 depth_mode = sl.DEPTH_MODE.ULTRA,
                                 coordinate_units = sl.UNIT.METER,
                                 coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime_parameters = sl.RuntimeParameters(sensing_mode = sl.SENSING_MODE.STANDARD)
        
    depth_image = sl.Mat()
    depth_cloud  = sl.Mat()
    frames = []
    timestamps = []

    t_start = time.time()
    t_end = 20
    t_current = 0
    i = 1

    while t_current <= t_end:
        print("Frame: " + str(i))
        timestamps.append(time.time())
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(depth_cloud, sl.MEASURE.XYZ)
            cloud = depth_cloud.get_data()

            # Drop unused forth dimension and reshape
            cloud = np.reshape(cloud, (-1, 4))
            cloud = cloud[~np.isnan(cloud[:,2]), :-1]
            cloud = cloud[np.isfinite(cloud[:,2]), :]

        if int(t_current) != int(time.time() - t_start):
            print("Time recorded: {} ...".format(int(t_current + 1)))
        t_current = time.time() - t_start
        i += 1

    zed.close()


    current_datetime = datetime.datetime.now()
    video_title = "../../logs/cloud_zed2_" + current_datetime.strftime("%y%m%d-%H%M%S")
    np.savez_compressed(video_title, data=frames, timestamp=timestamps)


if __name__ == "__main__":
    main()
