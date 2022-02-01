# Michel Heinemann
# calulate bias, precision and edge precision from depth data

import numpy as np
import math as m
import cv2
import tkinter as tk
from tkinter import filedialog
from crop_target.crop_target import CropTarget

# Define target
shape   = 'rectangle'
center  = np.array([[0.0], [0.0], [0.985]])    # Center of plane
size    = np.array([0.48, 0.48])               # (width, height) in m
angle   = np.radians(0.0)                      # In degrees
edge_width = 10
target  = CropTarget(shape, center, size, angle, edge_width)

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


def calculate_edge_precision(image):
    _, thresh = cv2.threshold(image,100,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]
    sorted_areas = np.sort(areas)
    cnt = contours[areas.index(sorted_areas[-1])] #the biggest contour
    
    epsilon = 0.15*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    print(approx)
    print('--------')
    cv2.polylines(image, [approx], True, 255, 1)
    cv2.imshow('image', image)
    cv2.waitKey(0)


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
    mask_bool = np.bool_(mask)

    bias_array = []
    precision_array = []
    for i in range(num_frames):
        depth_image = data[i,:,:,2].astype(np.int16)
        image_cropped = target.crop_to_target(depth_image, extrinsic_params, intrinsic_params)

        mean_depth = cv2.mean(image_cropped, mask)[0]/1000
        bias = center[2] - mean_depth
        bias_array.append(bias[0])

        diff_array= image_cropped[mask_bool]/1000 - mean_depth
        precision = m.sqrt(np.sum(np.multiply(diff_array, diff_array)))
        precision_array.append(precision)

        image_cropped_with_edges = target.crop_to_target(depth_image, extrinsic_params, intrinsic_params, True)
        display2 = (image_cropped_with_edges * (255.0 / np.max(image_cropped_with_edges))).astype(np.uint8)
        calculate_edge_precision(display2)
        
        
    total_bias = sum(bias_array) / len(bias_array)
    total_precision = sum(precision_array) / len(precision_array)

    print("Bias: {:0.3f}, Precision: {:0.3f}".format(total_bias, total_precision))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
