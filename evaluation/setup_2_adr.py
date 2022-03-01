import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from crop_target.crop_target import CropTarget
from common.constants import *
import math
import copy
import matplotlib.pyplot as plt

def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Numpy file", ".npz")]) #initialdir = '/media/michel/0621-AD85', 

    # Define target
    shape   = 'rectangle'
    center  = np.array([[0.0], [-0.07], [2.0 - OFFSET['orbbec']]]) #y = -0.07 orbbec 12
    size    = np.asarray(TARGET_SIZE) - REDUCE_TARGET
    angle   = np.radians(20.0)
    edge_width = 0
    target  = CropTarget(shape, center, size, angle, edge_width)

    eval_setup_2(file_path, target, shape, center, size, angle, edge_width)


def eval_setup_2(file_path, target, shape, center, size, angle, edge_width, show_mask=True):

    print("Opening file: ", file_path, "\n")
    print("Experiment configuration - Setup 2 (ADR)\nDistance:\t{:.3f}m\nTarget size:\t({:.3f},{:.3f})m\nAngle:\t\t{:.3f}rad\nEdge width:\t{}px".format(
         np.squeeze(center[2]), np.squeeze(size[0]), np.squeeze(size[1]), angle, edge_width))

    array = np.load(file_path)
    data  = array['data'][4:].astype(np.int16)
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

    target_reduced = copy.deepcopy(target)
    target_reduced.angle = np.sign(target.angle) * np.radians(60.0)

    num_frames = data.shape[0]
    image_dim = data[0,:,:,2].shape
    #mask = target.give_mask(image_dim, extrinsic_params, intrinsic_params)
    mask = target_reduced.give_adp_mask(image_dim, extrinsic_params, intrinsic_params)
    mask_bool = np.bool_(mask)

    valid_points_ratio = np.zeros((num_frames, 1))
    std = np.zeros((num_frames, 1))
    for i in range(num_frames):
        depth_image = data[i,:,:,2].astype(np.int16)
        image_cropped = target_reduced.crop_to_target_adp(depth_image, extrinsic_params, intrinsic_params)

        # cv2.imshow('reduced image', image_cropped)
        # cv2.waitKey(0)

        # plt.figure(1)
        # plt.imshow(image_cropped)
        # plt.show()

        mean = np.nanmean(image_cropped)
        #print(mean)
        
        not_nan = ~np.isnan(image_cropped)
        not_zero = image_cropped > 0
        not_toolarge = np.abs(image_cropped) < (mean + 1000)
        not_toosmall = np.abs(image_cropped) > (mean - 1000)
        valid_pixels = not_nan & not_zero & not_toolarge & not_toosmall

        # std[i] = np.std(image_cropped[valid_pixels])



        # not_nan = ~np.isnan(image_cropped[mask_bool])
        # not_zero = image_cropped[mask_bool] > 0
        # valid_pixels = not_nan & not_zero

        std_img = np.std(image_cropped[valid_pixels])

        width = target_reduced.size[1]*1000 * math.cos(np.deg2rad(60)) / math.cos(angle)
        # print(width)
        # print(image_cropped.shape)
        t = np.linspace(-width/2 * math.sin(angle), width/2 * math.sin(angle), num=image_cropped.shape[1])
        img_ideal = np.tile(t,(image_cropped.shape[0],1)).reshape(image_cropped.shape)

        #std_ideal = np.std(t)

        # print(std_ideal)
        # print(std_img)
        # std_rows = np.zeros((image_cropped.shape[1],1))
        # for j in range(image_cropped.shape[1]):
        #     std_rows[j] = np.std(image_cropped[:,j]-t)

        std[i] = np.std(abs(image_cropped-img_ideal))
        #valid_points_ratio[i] = (not_nan & not_zero).sum() / mask_bool.sum()

    #total_adr = np.mean(valid_points_ratio)
    total_std = np.mean(std)
    print("Angle-dependend precision: {:0.3f}".format(total_std))

    cv2.destroyAllWindows()

    return total_std, first_image_with_target


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
        print("Mask applied correctly")
        cv2.destroyAllWindows()
        return True
    else:
        print("Mask applied incorrectly, please adjust target parameters...")
        return False


if __name__ == "__main__":
    main()
