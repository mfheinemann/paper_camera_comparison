import os
import time
import pc_registration.reglib as reglib

import tkinter as tk
from tkinter import filedialog

import numpy as np

from common.constants import *
from crop_target.crop_target import CropTarget


# Define target
shape   = 'rectangle'
center  = np.array([[-0.029], [0.0], [1.0 - OFFSET['rs455']]])
size    = np.array([0.35, 0.20])
angle   = 0.0
edge_width = EDGE_WIDTH
target  = CropTarget(shape, center, size, angle, edge_width)


def main():
    # Only needed if you want to use manually compiled library code
    # reglib.load_library(os.path.join(os.curdir, "cmake-build-debug"))

    initial_guess = 1000

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Numpy file", ".npz")]) #initialdir = '/media/michel/0621-AD85', 

    print("Opening file: ", file_path, "\n")

    array = np.load(file_path)
    data  = array['data']
    extrinsic_params_data = array['extrinsic_params']
    intrinsic_params_data = array['intrinsic_params']
    extrinsic_params = extrinsic_params_data[0, :, :]
    intrinsic_params = intrinsic_params_data[0, :, :]

    point_cloud = data[15]

    point_cloud_cropped = target.crop_to_target(point_cloud, extrinsic_params, intrinsic_params)

    point_cloud_cropped[:,:,2] -= initial_guess

    point_cloud_cropped[:,:,1] = -point_cloud_cropped[:,:,1]

    num_points = point_cloud_cropped.shape[0] * point_cloud_cropped.shape[1]
    target_points = point_cloud_cropped.reshape(num_points, point_cloud_cropped.shape[2]).astype(np.float64)/10

    # Load you data
    source_points = reglib.load_data(os.path.join(os.curdir, "pc_registration/files", "euro_box.csv"))
    #target_points = reglib.load_data(os.path.join(os.curdir, "pc_registration/files", "scene_points.csv"))

    source_points = source_points

    # Run the registration algorithm
    start = time.time()
    trans = reglib.icp(source=source_points, target=target_points, nr_iterations=1, epsilon=0.5,
                       inlier_threshold=10, distance_threshold=10, downsample=0, visualize=True)
                       #resolution=12.0, step_size=0.5, voxelize=0)
    print("Runtime:", time.time() - start)
    print(trans)


if __name__ == "__main__":
    reglib.load_library(path = './pc_registration/reglib.py')
    main()