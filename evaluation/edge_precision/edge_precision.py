import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
from common.constants import *

def calculate_edge_precision(target, edge_masks, img_cnt, point_cloud_cropped, extrinsic_params, intrinsic_params):
    reg = LinearRegression()

    pc_list = point_cloud_cropped.reshape(point_cloud_cropped.shape[0]*point_cloud_cropped.shape[1], 3)

    results = np.zeros((1,1))
    for edge_mask in edge_masks:

        edge_mask_cropped = target.crop_to_target(edge_mask, extrinsic_params, intrinsic_params, True)
        img_cnt_edges = cv2.bitwise_and(img_cnt, img_cnt, mask=edge_mask_cropped)

        # Return non-zero pixel coordinates in mask
        pixel_z = np.argwhere(img_cnt_edges)
        #print(img_cnt_edges.shape)
        idxs = np.argwhere(img_cnt_edges.reshape(img_cnt_edges.shape[0]*img_cnt_edges.shape[1]))
        #print(idxs.shape)
        # print(pixel_z)
        # print(pixel_z.shape())
        # idxs = pixel_z.reshape(pixel_z.shape[0]*pixel_z.shape[1],1)

        # Find long side of edge
        X_var = np.var(pixel_z, axis=0)
        idx   = np.argmax(X_var)
        if idx == 0:
            long = 0
            short = 1

        else:
            long = 1
            short = 0


        edge_precision = np.min(np.std(pc_list[idxs, 0:2], axis = 0))

        # except: edge_precision = np.nan
        results = np.append(results,[[edge_precision]],axis=1) 

    return results[0,1:]


def get_edge_precision(target, point_cloud, mean_depth, extrinsic_params, intrinsic_params):    

    image = point_cloud[:,:,2]
    
    # Crop out target and set pixels further away than target to zero
    point_cloud_cropped = target.crop_to_target(point_cloud, extrinsic_params, intrinsic_params, True)

    image_cropped = point_cloud_cropped[:,:,2]

    target_mask   = cv2.inRange(image_cropped, 0.1,  mean_depth + DISTANCE_FRAME)

    contours, _ = cv2.findContours(target_mask.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    sorted_areas = np.sort(areas)
<<<<<<< HEAD
    #try:
    idx = areas.index(sorted_areas[-1])
=======

    try:
        idx = areas.index(sorted_areas[-1])
>>>>>>> 81347b1f05d70114c0dc31102d70fd0fed77c1a8


    # Draw contour
    img_cnt = np.zeros(image_cropped.shape, np.uint8)
    cv2.drawContours(img_cnt, contours, idx, (255),1)

<<<<<<< HEAD
    # Create mask of edge
    edge_masks = target.create_edge_masks(image.shape, extrinsic_params, intrinsic_params)

    edge_precision = calculate_edge_precision(target, edge_masks, img_cnt, point_cloud_cropped, extrinsic_params, intrinsic_params)
    #except: edge_precision = [np.nan, np.nan, np.nan, np.nan]
=======
        edge_precision = calculate_edge_precision(target, edge_masks, img_cnt, extrinsic_params, intrinsic_params)
    except: 
        edge_precision = [np.nan, np.nan, np.nan, np.nan]

>>>>>>> 81347b1f05d70114c0dc31102d70fd0fed77c1a8
    return edge_precision
