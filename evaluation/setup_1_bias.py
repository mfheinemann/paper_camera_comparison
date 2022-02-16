# Michel Heinemann
# calulate bias, precision and edge precision from depth data

import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from crop_target.crop_target import CropTarget
from edge_precision.edge_precision import *
from common.constants import *


def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Numpy file", ".npz")]) #initialdir = '/media/michel/0621-AD85', 

    # Define target
    shape   = 'rectangle'
    center  = np.array([[0.0], [0.0], [2.0 - OFFSET['oak-d']]])
    size    = np.asarray(TARGET_SIZE) - REDUCE_TARGET
    angle   = 0.0
    edge_width = EDGE_WIDTH
    target  = CropTarget(shape, center, size, angle, edge_width)

    eval_setup_1(file_path, target, shape, center, size, angle, edge_width)


def eval_setup_1(file_path, target, shape, center, size, angle, edge_width, show_mask=True):
    print("Opening file: ", file_path, "\n")
    print("Experiment configuration - Setup 1 (Bias, Precision)\nDistance:\t{:.3f}m\nTarget size:\t({:.3f},{:.3f})m\nAngle:\t\t{:.3f}rad\nEdge width:\t{}px".format(
         np.squeeze(center[2]), np.squeeze(size[0]), np.squeeze(size[1]), angle, edge_width))

    array = np.load(file_path)
    data  = array['data'][4:]
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
    mask_bool = np.bool_(mask)

    bias = np.zeros((num_frames, 1))
    precision = np.zeros((num_frames, 1))
    nan_ratio = np.zeros((num_frames, 1))
    edge_precision = np.zeros((num_frames, 4))
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
        not_toolarge = np.abs(image_cropped) < (center[2] + 2)
        not_toosmall = np.abs(image_cropped) > (center[2] - 2)
        valid_pixels = not_nan & not_zero & not_toolarge & not_toosmall
        mean_depth = cv2.mean(image_cropped[valid_pixels])[0]
        bias[i] = np.abs(center[2] - mean_depth)

        precision[i] = np.std(image_cropped[valid_pixels])

        nan_ratio[i] = (~valid_pixels).sum() / mask_bool.sum()

        # Correct target to real size to get edges
        target.size = np.asarray(TARGET_SIZE)
        edge_precision[i, :] = get_edge_precision(target, depth_image, mean_depth, extrinsic_params, intrinsic_params)
        target.size = size        
    
    total_bias = np.mean(bias)
    total_precision = np.mean(precision)
    total_nan_ratio = np.mean(nan_ratio)
    nan_edges = np.isnan(edge_precision).sum()
    nan_edge_ratio = nan_edges / (num_frames*4)
    total_edge_precision = np.nanmean(edge_precision, axis=0)

    print("Bias: {:0.5f}, Precision: {:0.5f}, at NaN-Ratio: {:0.5f}".format(total_bias, total_precision, total_nan_ratio))
    print("Edge Precision: {:0.5f} (left), {:0.5f} (down), {:0.5f} (right), {:0.5f} (up)".format(
        total_edge_precision[0], total_edge_precision[1],total_edge_precision[2], total_edge_precision[3]))

    cv2.destroyAllWindows()

    return total_bias, total_precision, total_nan_ratio, total_edge_precision, nan_edge_ratio, first_image_with_target


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


if __name__ == "__main__":
    main()
