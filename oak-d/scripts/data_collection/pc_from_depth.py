#!/usr/bin/env python3


import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import filedialog


def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Numpy file", ".npz")]) #initialdir = '/media/michel/0621-AD85', 
    depth_array_path = file_path[:-4] + '_fix'

    print(depth_array_path)

    array = np.load(file_path)
    data  = array['data']
    extrinsic_params_data = array['extrinsic_params']
    intrinsic_params_data = array['intrinsic_params']
    intrinsic_params = intrinsic_params_data[0, :, :]

    img_dim = data.shape[1:3]
    num_frames = data.shape[0]
    frames_array = np.zeros((num_frames, img_dim[0], img_dim[1], 3), dtype=np.uint16)
    for i in range(num_frames):
        print("Frame: ", i)
        depth_image = data[i,:,:,2].astype(np.int16)

        point_cloud = create_point_cloud(intrinsic_params, depth_image)
        frames_array[i,:,:] = point_cloud
   
    np.savez_compressed(depth_array_path, data=frames_array, 
                        intrinsic_params=intrinsic_params_data, 
                        extrinsic_params=extrinsic_params_data)
    
def create_point_cloud(in_params, depth_image):
    image_dim = depth_image.shape

    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(image_dim[0], image_dim[1], in_params[0,0], in_params[1,1], in_params[0,2], in_params[1,2])

    # PC form depth image
    pcl = o3d.geometry.PointCloud()
    pcl = pcl.create_from_depth_image(o3d.geometry.Image(depth_image), intr, project_valid_depth_only = False)

    # flip the orientation, so it looks upright, not upside-down
    pcl_points = np.asanyarray(pcl.points)
    point_cloud_array = np.int16(1000*pcl_points.reshape(image_dim[0], image_dim[1], 3))


    # For viualization
    # pcl = o3d.geometry.PointCloud()
    # pcl.points = o3d.utility.Vector3dVector(point_cloud_array.reshape(-1,3))
    # o3d.visualization.draw_geometries([pcl])


    return point_cloud_array

if __name__ == "__main__":
    main()
