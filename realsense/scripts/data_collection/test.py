import numpy as np
from datetime import datetime
import cv2
import tkinter as tk
from tkinter import filedialog

def main():
    root = tk.Tk()
    root.withdraw()

    file_path = '../../logs/log_rsd455_test_pc.npz'

    array = np.load(file_path)

    print(array["data"][5].shape)
    pc_array = np.zeros((3,480, 640))
    point_cloud_array = array["data"][5]

    pc = point_cloud_array.view(np.float32).reshape((point_cloud_array.size, 3))
    print(pc.shape)
    pc_x = pc[:,0].reshape((480, 640))
    pc_y = pc[:,1].reshape((480, 640))
    pc_z = pc[:,2].reshape((480, 640))

    pc2 = np.stack([pc_x, pc_y, pc_z], axis=0)

    pc3 = pc.reshape((480, 640, 3))

    # for k in range(2):
    #     for i in range(639):
    #         for j in range(479):
    #             pc_array[0,j,i] = point_cloud_array[j,i][0]


    # print(type(pc2))
    # print(pc2.shape)
    

    cv2.imshow("image", pc3[:, :, 2])
    # print(array2["1.408529281616211"].shape)
    # cv2.imshow("image_2", np.asanyarray(array2["1.408529281616211"]))
    cv2.waitKey(0)



if __name__ == "__main__":
    main()