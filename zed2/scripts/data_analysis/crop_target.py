import sys
import pyzed.sl as sl
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(initialdir="../../logs")

    array = np.load(file_path)
    data = array['data']
    cam_matrix = array['cam_matrix']
    image = data[0,:,:]
    cam_params = cam_matrix[0, :, :]

    # Camera matrix (here example from Zed2)
    # K = np.array([[cam_params.left_cam.fx,                      0, cam_params.left_cam.cx],
    #           [                     0, cam_params.left_cam.fy, cam_params.left_cam.cy],
    #           [                     0,                      0,                      1]])
    #K = np.array([[521.02642822, 0.0,          661.48553467],
    #             [0.0,         521.02642822, 360.91677856],
    #             [0.0,           0.0,           1.0        ]])
    #world2cam = np.array([[1, 0, 0, 1],[0, 1, 0, 1],[0, 0, 1, 1]])
    #cam_matrix = np.dot(K, world2cam)

    target1 = np.array([[-0.5, -0.5, 3, 1], [-0.5, 0.5, 3, 1],[0.5, 0.5, 3, 1],[0.5, -0.5, 3, 1]])
    target2 = np.array([[-0.5, -0.5, 6, 1], [-0.5, 0.5, 6, 1],[0.5, 0.5, 6, 1],[0.5, -0.5, 6, 1]])
    target3 = np.array([[-0.5, -0.5, 1, 1], [-0.5, 0.5, 1, 1],[0.5, 0.5, 3, 1],[0.5, -0.5, 3, 1]])

    point_2D = draw_rectangle(cam_params, target1, image)
    draw_rectangle(cam_params, target2, image)
    draw_rectangle(cam_params, target3, image)
    
    # draw mask
    mask = np.full((image.shape[0], image.shape[1]), 0, dtype=np.uint8)  # mask is only 
    cv2.rectangle(mask, point_2D[0],point_2D[2], (255, 255, 255), -1)
    

    disp = (image * (255.0 / np.max(image))).astype(np.uint8)
    disp = cv2.applyColorMap(image, cv2.COLORMAP_JET)

    # get first masked value (foreground)
    cropped = cv2.bitwise_and(disp, disp, mask=mask)


    cv2.imshow("ZED | 2D cropped", cropped)
    cv2.imshow("ZED | 2D Mask", mask)
    cv2.imshow("ZED | 2D View", disp)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def draw_rectangle(cam_matrix, target, image):
    point_2D = []

    for point_3D in target:
        point = np.matmul(cam_matrix, point_3D)
        point = point / point[2]
        point_2D.append(tuple(point[:-1].astype(int)))

    print(point_2D)
    cv2.line(image,point_2D[0],point_2D[1],(255,255,0),5)
    cv2.line(image,point_2D[1],point_2D[2],(255,255,0),5)
    cv2.line(image,point_2D[2],point_2D[3],(255,255,0),5)
    cv2.line(image,point_2D[3],point_2D[0],(255,255,0),5)

    return point_2D

if __name__ == "__main__":
    main()
