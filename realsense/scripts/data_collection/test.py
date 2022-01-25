import numpy as np
from datetime import datetime
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import sys

def main():
    root = tk.Tk()
    root.withdraw()

    if messagebox.askokcancel("Exit", "File already exists! Overwrite ?"):
        print("Overwriting!")
    else:
        print("Canceling!")
        sys.exit()


    file_path = '../../logs/log_rsd455_02_pc'

    npzfile = np.load(file_path)    
    print(len(npzfile.files))
    #timestamps_array = np.stack(timestamps, axis=0)
    print(len(npzfile.files))
    #timestamps_array = np.stack(timestamps, axis=0)
    frames_array = np.empty((len(npzfile.files), 720, 1280, 3), dtype=np.uint8)
    i = 0
    for key, value in npzfile.items():
        frames_array[i,:,:,:] = value

    keys, values = npzfile.values()

    frames_array = np.stack(values, axis=0)

    np.savez_compressed(file_path, data=frames_array)

    # print(array["data"][5].shape)
    # pc_array = np.zeros((3,480, 640))
    # point_cloud_array = array["data"][5]

    # pc = point_cloud_array.view(np.float32).reshape((point_cloud_array.size, 3))
    # print(pc.shape)
    # pc3 = pc.reshape((480, 640, 3))

    # for k in range(2):
    #     for i in range(639):
    #         for j in range(479):
    #             pc_array[0,j,i] = point_cloud_array[j,i][0]


    # print(type(pc2))
    # print(pc2.shape)
    

    # cv2.imshow("image", pc3[:, :, 2])
    # # print(array2["1.408529281616211"].shape)
    # # cv2.imshow("image_2", np.asanyarray(array2["1.408529281616211"]))
    # cv2.waitKey(0)



if __name__ == "__main__":
    main()