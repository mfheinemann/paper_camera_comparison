import sys
import ogl_viewer.viewer as gl
import pyzed.sl as sl
import cv2
import numpy as np

def main():
    print("Running Depth Sensing sample ... Press 'q' to quit")

    init = sl.InitParameters(camera_resolution = sl.RESOLUTION.HD720,
                                 depth_mode = sl.DEPTH_MODE.ULTRA,
                                 coordinate_units = sl.UNIT.METER,
                                 coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)

    zed = sl.Camera()
    pose = sl.Pose()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime_parameters = sl.RuntimeParameters(sensing_mode = sl.SENSING_MODE.STANDARD,
                                                confidence_threshold = 100)
    cam_params = zed.get_camera_information().calibration_parameters

    K = np.array([[cam_params.left_cam.fx,                      0, cam_params.left_cam.cx],
              [                     0, cam_params.left_cam.fy, cam_params.left_cam.cy],
              [                     0,                      0,                      1]])
    print(K)
    R = pose.get_rotation_matrix(sl.Rotation()).r.T     
    t = pose.get_translation(sl.Translation()).get()            
    world2cam = np.hstack((R, np.dot(-R, t).reshape(3,-1)))
    P = np.dot(K, world2cam)

    points1 = np.array([[-0.5, -0.5, 3, 1], [-0.5, 0.5, 3, 1],[0.5, 0.5, 3, 1],[0.5, -0.5, 3, 1]])
    points2 = np.array([[-0.5, -0.5, 6, 1], [-0.5, 0.5, 6, 1],[0.5, 0.5, 6, 1],[0.5, -0.5, 6, 1]])
    points3 = np.array([[-0.5, -0.5, 1, 1], [-0.5, 0.5, 1, 1],[0.5, 0.5, 3, 1],[0.5, -0.5, 3, 1]])

    point_2D = [] #np.zeros((4,1))
    depth_image = sl.Mat()
    i = 1
    while i < 300:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(depth_image, sl.VIEW.DEPTH)
            image = depth_image.get_data()

        point_2D = []  
        for point_3D in points1:
            point = np.matmul(P, point_3D)
            point = point / point[2]
            point_2D.append(tuple(point[:-1].astype(int)))

        cv2.line(image,point_2D[0],point_2D[1],(255,0,0),5)
        cv2.line(image,point_2D[1],point_2D[2],(255,0,0),5)
        cv2.line(image,point_2D[2],point_2D[3],(255,0,0),5)
        cv2.line(image,point_2D[3],point_2D[0],(255,0,0),5)
        
        # draw mask
        mask = np.full((image.shape[0], image.shape[1]), 0, dtype=np.uint8)  # mask is only 
        cv2.rectangle(mask, point_2D[0],point_2D[2], (255, 255, 255), -1)


        point_2D = []
        for point_3D in points2:
            point = np.matmul(P, point_3D)
            point = point / point[2]
            point_2D.append(tuple(point[:-1].astype(int)))

        cv2.line(image,point_2D[0],point_2D[1],(255,255,0),5)
        cv2.line(image,point_2D[1],point_2D[2],(255,255,0),5)
        cv2.line(image,point_2D[2],point_2D[3],(255,255,0),5)
        cv2.line(image,point_2D[3],point_2D[0],(255,255,0),5)

        point_2D = []  
        for point_3D in points3:
            point = np.matmul(P, point_3D)
            point = point / point[2]
            point_2D.append(tuple(point[:-1].astype(int)))

        cv2.line(image,point_2D[0],point_2D[1],(255,0,255),5)
        cv2.line(image,point_2D[1],point_2D[2],(255,0,255),5)
        cv2.line(image,point_2D[2],point_2D[3],(255,0,255),5)
        cv2.line(image,point_2D[3],point_2D[0],(255,0,255),5)

        
        # get first masked value (foreground)
        fg = cv2.bitwise_and(image, image, mask=mask)


        cv2.imshow("ZED | 2D fg", fg)
        cv2.imshow("ZED | 2D Mask", mask)
        cv2.imshow("ZED | 2D View", image)
        cv2.waitKey(5)

        i += 1

    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()
