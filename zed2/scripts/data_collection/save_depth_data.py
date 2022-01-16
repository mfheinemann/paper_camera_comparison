import ogl_viewer.viewer as gl
import pyzed.sl as sl
import time
import datetime
import numpy as np
import cv2

def main():
    init = sl.InitParameters(camera_resolution = sl.RESOLUTION.HD2K,
                                 camera_fps = 15,
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
    depth_map  = sl.Mat()
    frames = []
    timestamps = []

    t_start = time.time()
    t_end = 50
    t_current = 0
    i = 1

    while t_current <= t_end:
        print("Frame: " + str(i))
        timestamps.append(time.time())
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            zed.retrieve_image(depth_image, sl.VIEW.DEPTH)
            map = depth_map.get_data()
            image = depth_image.get_data()

            frames.append(np.float16(map))
            cv2.imshow("ZED | image", image)
            cv2.waitKey(1)
            
        if int(t_current) != int(time.time() - t_start):
            print("Time recorded: {} ...".format(int(t_current + 1)))
        t_current = time.time() - t_start
        i += 1

    cv2.destroyAllWindows()
    zed.close()

    timestamps_array = np.stack(timestamps, axis=0)
    frames_array = np.stack(frames, axis=0)

    current_datetime = datetime.datetime.now()
    video_title = "../../logs/log_zed2_" + current_datetime.strftime("%y%m%d%H%M%S")
    np.savez_compressed(video_title, data=frames_array, timestamp=timestamps_array)


if __name__ == "__main__":
    main()
