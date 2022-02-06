import numpy as np
from datetime import datetime
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import sys
import matplotlib
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

# def main():
#     # root = tk.Tk()
#     # root.withdraw()

#     # if messagebox.askokcancel("Exit", "File already exists! Overwrite ?"):
#     #     print("Overwriting!")
#     # else:
#     #     print("Canceling!")
#     #     sys.exit()


#     file_path = '../../logs/log_rsd455_test_pc.npz'

#     # file_path = '../../../zed2/logs/log_zed2_2_pc.npz'

#     npzfile = np.load(file_path) 


#     data = npzfile["data"]
#     print(data[100, 340:380, 620:660, 2])

#     depth_image=data[100, :, :, 2].astype(np.int16)

#     cv2.imshow('image', depth_image)

#     # plt.figure(1)
#     # plt.imshow(depth_image)
#     # plt.show()

#     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

#     # depth_colormap = (depth_image * (255.0 / np.max(depth_image))).astype(np.uint8)
#     # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

#     cv2.imshow('image color',depth_colormap)
#     if cv2.waitKey(0) == ord("q"):
#         return


#     # print(np.uint16(-519))
#     # print(np.int16(np.uint16(-519)))


#     # print(len(npzfile.files))
#     #timestamps_array = np.stack(timestamps, axis=0)
#     # print(len(npzfile.files))
#     #timestamps_array = np.stack(timestamps, axis=0)
#     # frames_array = np.empty((len(npzfile.files), 720, 1280, 3), dtype=np.uint8)
#     # i = 0
#     # for key, value in npzfile.items():
#     #     frames_array[i,:,:,:] = value

#     # keys, values = npzfile.values()

#     # frames_array = np.stack(values, axis=0)

#     # np.savez_compressed(file_path, data=frames_array)

#     # print(array["data"][5].shape)
#     # pc_array = np.zeros((3,480, 640))
#     # point_cloud_array = array["data"][5]

#     # pc = point_cloud_array.view(np.float32).reshape((point_cloud_array.size, 3))
#     # print(pc.shape)
#     # pc3 = pc.reshape((480, 640, 3))

#     # for k in range(2):
#     #     for i in range(639):
#     #         for j in range(479):
#     #             pc_array[0,j,i] = point_cloud_array[j,i][0]


#     # print(type(pc2))
#     # print(pc2.shape)
    

#     # cv2.imshow("image", pc3[:, :, 2])
#     # # print(array2["1.408529281616211"].shape)
#     # # cv2.imshow("image_2", np.asanyarray(array2["1.408529281616211"]))
#     # cv2.waitKey(0)



# if __name__ == "__main__":
#     main()
reg = LinearRegression()
pixel = np.array([[0,1],[0,2],[0,3],[0,4],[0,5]])
# pixel = np.array([pixel[:,1],pixel[:,0]])
pixel = np.array([pixel[:,0],pixel[:,1]])

# Find long side of edge
X_var = np.var(pixel, axis=0)
idx   = np.argmax(X_var)
if idx == 0:
    sort_idx = np.argsort(pixel[:,0], axis=0)
    X = pixel[sort_idx,0].reshape(-1, 1)
    y = pixel[sort_idx,1]
else:
    sort_idx = np.argsort(pixel[:,1], axis=0)
    X = pixel[sort_idx,1].reshape(-1, 1)
    y = pixel[sort_idx,0]

reg.fit(X, y)
y_pred = reg.predict(X)

print(X)
print(y)
print(y_pred)