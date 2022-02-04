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
    image_cropped = target.crop_to_target(image, extrinsic_params, intrinsic_params, True)
    target_mask   = cv2.inRange(image_cropped, 0.1,  mean_depth + 0.2)

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
