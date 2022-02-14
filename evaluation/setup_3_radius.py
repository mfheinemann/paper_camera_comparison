import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from crop_target.crop_target import CropTarget
import open3d as o3d
from common.constants import *

def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Numpy file", ".npz")]) #initialdir = '/media/michel/0621-AD85', 

    # Define target
    shape   = 'circle'
    center  = np.array([[0.0], [0.0], [2.0 - OFFSET['zed2']]])
    size    = SPHERE_RADIUS
    angle   = 0.0
    edge_width = 0
    target  = CropTarget(shape, center, size, angle, edge_width)

    eval_setup_3_1(file_path, target, shape, center, size, angle, edge_width)


def eval_setup_3_1(file_path, target, shape, center, size, angle, edge_width, show_mask=True):

    print("Opening file: ", file_path, "\n")
    print("Experiment configuration - Setup 3 (RRE)\nDistance:\t{:.3f}m\nTarget radius:\t{:.3f}m\nAngle:\t\t{:.3f}rad\nEdge width:\t{}px".format(
         np.squeeze(center[2]), size, angle, edge_width))

    array = np.load(file_path)
    data  = array['data']
    extrinsic_params_data = array['extrinsic_params']
    intrinsic_params_data = array['intrinsic_params']
    extrinsic_params = extrinsic_params_data[0, :, :]
    intrinsic_params = intrinsic_params_data[0, :, :]

    depth_image = data[5,:,:,2].astype(np.int16)

    disp = (depth_image * (255.0 / np.max(depth_image))).astype(np.uint8)
    disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

    first_image_with_target = target.show_target_in_image(disp, extrinsic_params, intrinsic_params)

    if show_mask:
        is_mask_correct = prepare_images(data, target, extrinsic_params, intrinsic_params)
        if is_mask_correct == False:
            return

    num_frames = data.shape[0]
    image_dim = data[0,:,:,2].shape
    mask = target.give_mask(image_dim, extrinsic_params, intrinsic_params)
    mask_bool = np.squeeze(np.bool_(mask))

    radius_mean = np.zeros((num_frames, 1))
    radius_std = np.zeros((num_frames, 1))
    for i in range(num_frames):
        # Load point cloud and transform millimeter value to meter
        point_cloud = data[i,:,:].astype(np.int16) / 1000
        point_cloud_cropped = target.crop_to_target(point_cloud, extrinsic_params, intrinsic_params)

        radius_mean[i], radius_std[i] = get_radius_estimate(point_cloud_cropped[mask_bool], center)

    total_radius_mean = np.mean(radius_mean) - size
    total_radius_std = np.mean(radius_std)
    print("Radius Reconstruction Error: {:0.3f} (mean), {:0.3f} (std)".format(total_radius_mean, total_radius_std))

    cv2.destroyAllWindows()

    return total_radius_mean, total_radius_std, first_image_with_target


def prepare_images(data, target, extrinsic_params, intrinsic_params):
    depth_image = data[0,:,:,2].astype(np.int16)

    disp = (depth_image * (255.0 / np.max(depth_image))).astype(np.uint8)
    disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

    image_dim = depth_image.shape
    target_cropped  = target.crop_to_target(disp, extrinsic_params, intrinsic_params)
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


def get_radius_estimate(point_cloud, center):
    # Remove Inf, NaN and zero values
    is_finite_depth = np.isfinite(point_cloud[:,2])
    is_not_zero = point_cloud[:,2] > 0
    is_not_background = point_cloud[:,2] < (center[2] + 0.3)
    selection_matrix = is_finite_depth & is_not_zero & is_not_background

    valid_points = point_cloud[selection_matrix, 0:3]

    radius = np.sqrt((valid_points[:,0] - center[0])**2 + 
                     (valid_points[:,1] - center[1])**2 + 
                     (valid_points[:,2] - center[2])**2)
    
    #points_vis = np.append(points, center.T, axis=0)
    #pcl_vis = o3d.geometry.PointCloud()
    #pcl_vis.points = o3d.utility.Vector3dVector(points_vis)
    #o3d.visualization.draw_geometries([pcl_vis])

    return np.mean(radius), np.std(radius)


if __name__ == "__main__":
    main()
