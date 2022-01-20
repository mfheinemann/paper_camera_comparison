import sys
import ogl_viewer.viewer as gl
import pyzed.sl as sl
import cv2

def main():
    print("Running Depth Sensing sample ... Press 'q' to quit")

    init = sl.InitParameters(camera_resolution = sl.RESOLUTION.HD720,
                                 depth_mode = sl.DEPTH_MODE.ULTRA,
                                 coordinate_units = sl.UNIT.METER,
                                 coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime_parameters = sl.RuntimeParameters(sensing_mode = sl.SENSING_MODE.STANDARD,
                                                confidence_threshold = 100)

    res = sl.Resolution()
    res.width = 720
    res.height = 404
    
    camera_model = zed.get_camera_information().camera_model
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(len(sys.argv), sys.argv, camera_model, res)
    
    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    depth_image = sl.Mat()

    while viewer.is_available():
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:

            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, res)
            viewer.updateData(point_cloud)
        
            zed.retrieve_image(depth_image, sl.VIEW.DEPTH)
            image = depth_image.get_data()
            cv2.imshow("ZED | 2D View", image)
            cv2.waitKey(5)

    cv2.destroyAllWindows()
    viewer.exit()
    zed.close()

if __name__ == "__main__":
    main()
