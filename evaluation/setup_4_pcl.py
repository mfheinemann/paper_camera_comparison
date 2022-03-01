from re import T
import tkinter as tk
from tkinter import filedialog
import numpy as np
import open3d as o3d

from common.constants import *
from crop_target.crop_target import CropTarget
from pc_registration.pc_utils import *

# Define target
shape   = 'rectangle'
center  = np.array([[-0.029], [0.0], [1.0 - OFFSET['rsd455']]])
size    = np.array([0.35, 0.20])
angle   = 0.0
edge_width = EDGE_WIDTH
target  = CropTarget(shape, center, size, angle, edge_width)


def main():
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

    point_cloud = data[15,:,:,:3].astype(np.int16)/1000
    point_cloud_cropped = target.crop_to_target(point_cloud, extrinsic_params, intrinsic_params)


    num_points = point_cloud_cropped.shape[0] * point_cloud_cropped.shape[1]
    target_points_np = point_cloud_cropped.reshape(num_points,-1).astype(np.float64)

    # Data manipulation for correct alignment
    target_points_np[:,1] *= -1.0

    target_points = o3d.geometry.PointCloud()
    target_points.points = o3d.utility.Vector3dVector(target_points_np)

    # Load you data
    source_points_np = load_data("pc_registration/files/euro_box.csv")/100
    source_points = o3d.geometry.PointCloud()
    source_points.points = o3d.utility.Vector3dVector(source_points_np)

    # Run the registration algorithm
    threshold = 0.2
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    draw_registration_result(source_points, target_points, trans_init)
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(source_points, target_points,
                                                       threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_points, target_points, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    draw_registration_result(source_points, target_points, reg_p2p.transformation)


if __name__ == "__main__":
    main()