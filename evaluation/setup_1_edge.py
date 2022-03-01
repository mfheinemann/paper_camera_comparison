# Michel Heinemann
# calulate bias, precision and edge precision from depth data

import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from crop_target.crop_target import CropTarget
from edge_precision.edge_precision import *
from common.constants import *
import scipy
from scipy import optimize
import copy


def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Numpy file", ".npz")]) #initialdir = '/media/michel/0621-AD85', 

    # Define target
    shape   = 'rectangle'
    center  = np.array([[0.0], [0.0], [5.0 - OFFSET['orbbec']]])
    size    = np.asarray(TARGET_SIZE) - REDUCE_TARGET
    angle   = 0.0
    edge_width = EDGE_WIDTH
    target  = CropTarget(shape, center, size, angle, edge_width)

    eval_setup_1_2(file_path, target, shape, center, size, angle, edge_width)


def eval_setup_1_2(file_path, target, shape, center, size, angle, edge_width, show_mask=True):
    print("Opening file: ", file_path, "\n")
    print("Experiment configuration - Setup 1 (Edge Precision)\nDistance:\t{:.3f}m\nTarget size:\t({:.3f},{:.3f})m\nAngle:\t\t{:.3f}rad\nEdge width:\t{}px".format(
         np.squeeze(center[2]), np.squeeze(size[0]), np.squeeze(size[1]), angle, edge_width))

    array = np.load(file_path)
    data  = array['data'][4:]
    extrinsic_params_data = array['extrinsic_params']
    intrinsic_params_data = array['intrinsic_params']
    extrinsic_params = extrinsic_params_data[0, :, :]
    intrinsic_params = intrinsic_params_data[0, :, :]

    data = data[:,:,:,0:3]

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
    mask_bool = np.bool_(mask)

    bias = np.zeros((num_frames, 1))
    precision = np.zeros((num_frames, 1))
    nan_ratio = np.zeros((num_frames, 1))

    edge_precision = np.zeros((num_frames, 1))
    target_large = copy.deepcopy(target)
    target_large.size += 0.05
    start = np.array([target.center[0], target.center[1]])/1000

    means = np.mean(data.astype(np.int16), axis=0)
    means_cropped = target_large.crop_to_target(means, extrinsic_params, intrinsic_params)

    print("optimizazion:")
    opt = scipy.optimize.minimize(get_edge_mean_error, start, args=(target, means_cropped), method='nelder-mead', options={"maxiter" : 30, "fatol": 0.50})
    opt_pos = opt.x
    print(opt_pos)

    # error_0 = get_edge_mean_error(target.center*1000, target, means_cropped)

    # print(error_0)

    # error_1 = get_edge_mean_error([opt_pos[0], opt_pos[1]], target, means_cropped)

    # print(error_1)

    for i in range(num_frames):
        depth_image = data[i,:,:,2].astype(np.int16)/1000
        image_cropped = target.crop_to_target(depth_image, extrinsic_params, intrinsic_params)

        # plt.figure(1)
        # plt.imshow(image_cropped)
        # plt.show()

        # disp = (depth_image * (255.0 / np.max(depth_image))).astype(np.uint8)
        # disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

        # cv2.imshow('image', disp)
        # cv2.waitKey(0)

        not_nan = ~np.isnan(image_cropped)
        not_zero = image_cropped > 0
        valid_pixels = not_nan & not_zero
        mean_depth = cv2.mean(image_cropped[valid_pixels])[0]

        bias[i] = np.abs(center[2] - mean_depth)

        precision[i] = np.std(image_cropped[valid_pixels])

        nan_ratio[i] = (~valid_pixels).sum() / mask_bool.sum()

        # Correct target to real size to get edges

        point_cloud = data[i,:,:,:].astype(np.int16)
        point_cloud_cropped = target_large.crop_to_target(point_cloud, extrinsic_params, intrinsic_params)

        # cv2.imshow('depth image', point_cloud_cropped[:,:,2])
        # cv2.waitKey(0)

        # print('calc:')
        edge_precision[i] = get_edge_mean_error([opt_pos[0], opt_pos[1]], target, point_cloud_cropped)
        #target.size = np.asarray(TARGET_SIZE)
        # edge_precision[i, :] = get_edge_precision(target, depth_image, mean_depth, extrinsic_params, intrinsic_params)
        #target.size = size        
    
    total_bias = np.mean(bias)
    total_precision = np.mean(precision)
    total_nan_ratio = np.mean(nan_ratio)
    nan_edges = np.isnan(edge_precision).sum()
    nan_edge_ratio = nan_edges / (num_frames*4)
    total_edge_precision = np.nanmean(edge_precision, axis=0)

    print("Bias: {:0.5f}, Precision: {:0.5f}, at NaN-Ratio: {:0.5f}".format(total_bias, total_precision, total_nan_ratio))
    print(total_edge_precision)
    print("Edge Precision: {:0.5f} ".format(total_edge_precision[0]))
    # print("Edge Precision: {:0.5f} (left), {:0.5f} (down), {:0.5f} (right), {:0.5f} (up)".format(
    #     total_edge_precision[0], total_edge_precision[1],total_edge_precision[2], total_edge_precision[3]))

    cv2.destroyAllWindows()

    return total_edge_precision, first_image_with_target


def prepare_images(data, target, extrinsic_params, intrinsic_params):
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
        print("Mask applied correctly\n")
        cv2.destroyAllWindows()
        return True
    else:
        print("Mask applied incorrectly, please adjust target parameters...")
        return False

def get_edge_mean_error(center, target, point_cloud):

    # find points on sphere
    num_points = point_cloud.shape[0] * point_cloud.shape[1]
    points = point_cloud.reshape(num_points, point_cloud.shape[2])

    depth_image = point_cloud[:,:,2]

    # get edge indexes
    idxs = get_edge_idxs(depth_image, target)
    #print(idxs.shape)

    edges = points[idxs[:,0],:].reshape([idxs.shape[0],3])
    # print(sphere)

    error = 0.0
    for i in range(idxs.shape[0]):
        x = edges[i,0]
        y = edges[i,1]
        z = edges[i,2]
        point = np.array([x, y, z])
        err = edge_error(point, center[0], center[1], target.center[2]*1000, target)/idxs.shape[0]
        # print(err)
        error += err

    # print(esrror)

    return error

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

    # print([e1, e2, e3, e4])

    error = min(e1, e2, e3, e4)
    # print(error)
    return error


def get_edge_idxs(image, target):

    not_nan = ~np.isnan(image)
    not_zero = image > 0
    valid_pixels = not_nan & not_zero
    mean_depth = cv2.mean(image[valid_pixels])[0]

    target_mask = cv2.inRange(image, 0.1,  mean_depth + 0.2)
    edges = cv2.Canny(target_mask.astype(np.uint8), 100, 200)

    edges[target.edge_width*2:edges.shape[0]-target.edge_width*2, target.edge_width*2:edges.shape[1]-target.edge_width*2] = 0.0

    idxs = np.argwhere(edges.reshape([edges.shape[0]*edges.shape[1], 1]))

    return idxs

if __name__ == "__main__":
    main()
