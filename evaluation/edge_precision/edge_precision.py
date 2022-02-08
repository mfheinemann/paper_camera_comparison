import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

def calculate_edge_precision(target, edge_masks, img_cnt, extrinsic_params, intrinsic_params):
    reg = LinearRegression()

    results = np.zeros((1,1))
    for edge_mask in edge_masks:

        edge_mask_cropped = target.crop_to_target(edge_mask, extrinsic_params, intrinsic_params, True)
        img_cnt_edges = cv2.bitwise_and(img_cnt, img_cnt, mask=edge_mask_cropped)


        #plt.figure(1)
        #plt.imshow(img_cnt)
        #plt.show()
        #cv2.imshow("edge", img_cnt_edges)
        #cv2.waitKey(0)

        # Return non-zero pixel coordinates in mask
        pixel = np.argwhere(img_cnt_edges)

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

        #plt.scatter(X, y,color='g')
        #plt.plot(X, y_pred,color='k')
        #plt.plot(X, np.around(y_pred),color='b')
        #plt.show()

        line_pixel_diff = np.abs(y - y_pred)
        edge_precision = np.std(line_pixel_diff)
        results = np.append(results,[[edge_precision]],axis=1) 

    return results[0,1:]


def get_edge_precision(target, image, mean_depth, extrinsic_params, intrinsic_params):
    # Crop out target and set pixels further away than target to zero
    target_mask   = cv2.inRange(image, 0.1,  mean_depth + 0.2)
    edges = cv2.Canny(target_mask.astype(np.uint8), 100, 200)
    image_cropped = target.crop_to_edge(edges, extrinsic_params, intrinsic_params, True)

    cv2.imshow("image", image_cropped)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(target_mask.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    sorted_areas = np.sort(areas)
    idx = areas.index(sorted_areas[-1])

    # Draw contour
    img_cnt = np.zeros(image_cropped.shape, np.uint8)
    cv2.drawContours(img_cnt, contours, idx, (255),1)

    # Create mask of edge
    edge_masks = target.create_edge_masks(image.shape, extrinsic_params, intrinsic_params)

    edge_precision = calculate_edge_precision(target, edge_masks, img_cnt, extrinsic_params, intrinsic_params)
   
    return edge_precision


def edge_error(point, x, y, z, target):
    x_dim = target.size[0]/2
    y_dim = target.size[1]/2
    c1 = [x+x_dim, y+y_dim, z]
    c2 = [x+x_dim, y-y_dim, z]
    c3 = [x-x_dim, y-y_dim, z]
    c4 = [x-x_dim, y+y_dim, z]
    e1 = abs((c2[0]-c1[0])*(c1[1]-point[1])-(c1[0]-point[0])*(c2[1]-c1[1]))/np.sqrt(np.square(c2[0]-c1[0])+np.square(c2[1]-c1[1]))
    e2 = abs((c3[0]-c2[0])*(c2[1]-point[1])-(c2[0]-point[0])*(c3[1]-c2[1]))/np.sqrt(np.square(c3[0]-c2[0])+np.square(c3[1]-c2[1]))
    e3 = abs((c4[0]-c3[0])*(c3[1]-point[1])-(c3[0]-point[0])*(c4[1]-c3[1]))/np.sqrt(np.square(c4[0]-c3[0])+np.square(c4[1]-c3[1]))
    e4 = abs((c1[0]-c4[0])*(c4[1]-point[1])-(c4[0]-point[0])*(c1[1]-c4[1]))/np.sqrt(np.square(c1[0]-c4[0])+np.square(c1[1]-c4[1]))

    error = min(e1, e2, e3, e4)
    return error

def image_error(center, point_cloud):
    # find points on edge
    num_points = point_cloud.shape[0] * point_cloud.shape[1]
    points = point_cloud.reshape(num_points, point_cloud.shape[2])

    idxs = np.argwhere(points != 0.0)

    edges = points[idxs].reshape(idxs.shape[0])

    error = 0.0
    for i in range(idxs.shape[0]):
        x = edges[i]
        y = edges[i]
        z = edges[i]
        point = np.array([x, y, z])
        error += edge_error(point, center[0], center[1], center[2], target)/idxs.shape[0]

    print(error)

    return error


def optimize_center(target, depth_image, mean_depth, extrinsic_params, intrinsic_params):
    image_cropped = get_edges(target, depth_image, mean_depth, extrinsic_params, intrinsic_params)

    # find points on sphere
    num_points = depth_image.shape[0] * depth_image.shape[1]
    points = depth_image.reshape(num_points)

    idxs = np.argwhere(points != 0.0)

    edges = points[idxs].reshape(idxs.shape[0])

    error = 0.0
    for i in range(idxs.shape[0]):
        x = edges[i]
        y = edges[i]
        z = edges[i]
        point = np.array([x, y, z])
        error += edge_error(point, center[0], center[1], center[2], target)/idxs.shape[0]

    print(error)

    return error



def get_edges(target, image, mean_depth, extrinsic_params, intrinsic_params):
    # Crop out target and set pixels further away than target to zero
    target_mask   = cv2.inRange(image, 0.1,  mean_depth + 0.2)
    edges = cv2.Canny(target_mask.astype(np.uint8), 100, 200)
    image_cropped = target.crop_to_edge(edges, extrinsic_params, intrinsic_params, True)
    return image_cropped