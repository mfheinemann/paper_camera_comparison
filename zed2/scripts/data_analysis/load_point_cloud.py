import numpy as np
from datetime import datetime
import cv2
import pyransac3d as pyrsc
import tkinter as tk
from tkinter import filedialog

def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(initialdir="../../logs")

    array = np.load(file_path)
    data = array['data']
    print(data.shape)
    depth_images = data[:,:,:,2]
    dim_depth_image = depth_images.shape

    for i in range(dim_depth_image[0]):
        disp = (depth_images[i, :, :] * (255.0 / np.max(depth_images[i, :, :]))).astype(np.uint8)
        disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
        cv2.imshow("ZED | 2D View", disp)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()
