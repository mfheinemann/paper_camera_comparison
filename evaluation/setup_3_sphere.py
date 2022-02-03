import numpy as np
import math as m
import cv2
import tkinter as tk
from tkinter import filedialog
from crop_target.crop_target import CropTarget
import open3d as o3d
import pyransac3d as pyrsc

# Define target
shape   = 'circle'
center  = np.array([[0.0], [0.0], [0.985]])    # Center of shperec
size    = 0.139 / 2.0                          # Radius in m
angle   = np.radians(0.0)
edge_width = 0
target  = CropTarget(shape, center, size, angle, edge_width)


def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Numpy file", ".npz")])

    array = np.load(file_path)
    data  = array['data']
    extrinsic_params_data = array['extrinsic_params']
    intrinsic_params_data = array['intrinsic_params']
    extrinsic_params = extrinsic_params_data[0, :, :]
    intrinsic_params = intrinsic_params_data[0, :, :]

    is_mask_correct = prepare_images(data, extrinsic_params, intrinsic_params)
    if is_mask_correct == False:
        return

    num_frames = data.shape[0]
    image_dim = data[0,:,:,2].shape
    mask = target.give_mask(image_dim, extrinsic_params, intrinsic_params)
    mask_bool = np.squeeze(np.bool_(mask))

    radius = np.zeros((num_frames, 1))
    for i in range(num_frames):
        point_cloud = data[i,:,:].astype(np.int16)
        point_cloud_cropped = target.crop_to_target(point_cloud, extrinsic_params, intrinsic_params)

        _, radius[i], _ = get_sphere_estimate(point_cloud_cropped[mask_bool])


    sphere_rec_error = np.abs(np.mean(radius) - size/2)
    print("Sphere Reconstruction Error: {:0.3f}".format(sphere_rec_error))

    cv2.destroyAllWindows()


def prepare_images(data, extrinsic_params, intrinsic_params):
    depth_image = data[0,:,:,2].astype(np.int16)

    disp = (depth_image * (255.0 / np.max(depth_image))).astype(np.uint8)
    disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

    image_dim = depth_image.shape
    target_cropped  = target.crop_to_target(disp, extrinsic_params, intrinsic_params, True)
    image_mask      = target.give_mask(image_dim, extrinsic_params, intrinsic_params)
    image_frame     = target.show_target_in_image(disp, extrinsic_params, intrinsic_params)

    cv2.imshow("Image cropped with extended edges", target_cropped)
    cv2.imshow("Mask", image_mask)
    cv2.imshow("Image with target frame", image_frame)

    print("If mask is applied correctly, press 'q'")
    key = cv2.waitKey(0)
    if key == 113:
        print("Mask applied correctly")
        cv2.destroyAllWindows()
        return True
    else:
        print("Mask applied incorrectly, please adjust target parameters...")
        return False


def get_sphere_estimate(point_cloud):
    # Remove Inf, NaN and zero values
    is_finite_depth = np.isfinite(point_cloud[:,2])
    is_not_zero = point_cloud[:,2] > 0

    points = point_cloud[is_finite_depth & is_not_zero, 0:2]

    sph = pyrsc.Sphere()
    center, radius, inliers = sph.fit(points, thresh=0.2, maxIteration=1000)

    #pcl_vis = o3d.geometry.PointCloud()
    #pcl_vis.points = o3d.utility.Vector3dVector(points)
    #inline = pcl_vis.select_by_index(inliers).paint_uniform_color([0, 1, 0])
    #outline = pcl_vis.select_by_index(inliers, invert=True).paint_uniform_color([1, 0, 0])

    #mesh_circle = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    #mesh_circle.compute_vertex_normals()
    #mesh_circle.paint_uniform_color([0.9, 0.1, 0.1])
    #mesh_circle = mesh_circle.translate((center[0], center[1], center[2]))
    #o3d.visualization.draw_geometries([outline, mesh_circle, inline])

    return center, radius, inliers


if __name__ == "__main__":
    main()
