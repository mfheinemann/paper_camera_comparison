import numpy as np
import open3d as o3d
import cv2
import math as m

file_path = '../../logs/log_oak-d_2_pc.npz'

array = np.load(file_path)
data  = array['data']
extrinsic_params_data = array['extrinsic_params']
intrinsic_params_data = array['intrinsic_params']
extrinsic_params = extrinsic_params_data[0, :, :]
intrinsic_params = intrinsic_params_data[0, :, :]

for i in range(data.shape[0]):
    pc_array = data[i,:,:,:]
    depth_image = pc_array[:,:,2]
    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(1280, 720, intrinsic_params[0,0], intrinsic_params[1,1], intrinsic_params[0,2], intrinsic_params[1,2])
    print(depth_image[100,100])
    # PC form depth image
    pcl = o3d.geometry.PointCloud()
    pcl = pcl.create_from_depth_image(o3d.geometry.Image((depth_image).astype(np.uint16)), intr, project_valid_depth_only = True)

    # flip the orientation, so it looks upright, not upside-down
    rot_angle = np.radians(180.0)
    pcl.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]])
    # pcl.transform([[m.cos(rot_angle),-m.sin(rot_angle),0,0],
    #                 [m.sin(rot_angle),m.cos(rot_angle),0,0],
    #                 [0,0,1,0],
    #                 [0,0,0,1]])
    # pcl.transform([[m.cos(rot_angle),0,m.sin(rot_angle),0],
    #                 [0,1,0,0],
    #                 [-m.sin(rot_angle),0, m.cos(rot_angle),0],
    #                 [0,0,0,1]])    

    # pcl_points = np.asanyarray(pcl.points)
    
    # cv2.imshow("depth_image", depth_image)
    # cv2.waitKey(0)


    pc_points = pc_array.reshape(1280*720, 3)

    # visualize point cloud
       
    o3d.visualization.draw_geometries([pcl])

    print(i)
