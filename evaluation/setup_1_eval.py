# Michel Heinemann
# calulate bias, precision and edge precision from depth data

import numpy as np
import math as m
import cv2
import tkinter as tk
from tkinter import filedialog
from crop_target.crop_target import CropTarget


# Define target
target = CropTarget()
shape   = 'rectangle'
if shape == 'rectangle':
    center  = np.array([[0.0], [0.0], [1.985]])    # Center of plane
    size    = np.array([0.48, 0.48])               # (width, height) in m
    angle   = np.radians(0.0)                     # In degrees
elif shape == 'circle':
    center  = np.array([[1.0], [0.0], [3.0]])   # Center of shpere
    size    = 0.2                               # Radius in m
    angle   = np.radians(0.0) 
else:
    print("Not a valid shape!")
edge_width  = 5


def prepare_images(data, extrinsic_params, intrinsic_params):
    depth_image = data[0,:,:,2]
    disp = (depth_image * (255.0 / np.max(depth_image))).astype(np.uint8)
    disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    image_mask = target.give_cropped_image(disp, extrinsic_params, intrinsic_params,
                                            shape, center, size, angle)
    image_frame = target.show_target_in_image(disp, extrinsic_params, intrinsic_params,
                                            shape, center, size, angle)

    edge_masks = target.create_edge_masks(disp, extrinsic_params, intrinsic_params,
                                            shape, center, size, angle, edge_width)
    all_edge_masks = cv2.bitwise_or(edge_masks[0], edge_masks[1])
    all_edge_masks = cv2.bitwise_or(all_edge_masks, edge_masks[2])
    all_edge_masks = cv2.bitwise_or(all_edge_masks, edge_masks[3])

    img_edge_mask = cv2.bitwise_and(disp, disp, mask=all_edge_masks)
    cv2.imshow("Mask on image", image_mask)
    cv2.imshow("Image with target frame", image_frame)
    cv2.imshow("Edge masks", img_edge_mask)

    print("If mask is applied correctly, press 'q'")
    key = cv2.waitKey(0)
    if key == 113:
        print("Mask applied correctly")
        cv2.destroyAllWindows()
        return True
    else:
        print("Mask applied incorrectly, please adjust target parameters...")
        return False

def run_canny():
    '''Run Canny'''


def calculate_edge_precision(image, edge, extrinsic_params, intrinsic_params):
    edge_masks = target.create_edge_masks(image, extrinsic_params, intrinsic_params,
                                        shape, center, size, angle, edge_width)
    if edge == 'left':
        img_edge_mask = cv2.bitwise_and(image, image, mask=edge_masks[0])
    elif edge == 'up':
        img_edge_mask = cv2.bitwise_and(image, image, mask=edge_masks[1])
    elif edge == 'right':
        img_edge_mask = cv2.bitwise_and(image, image, mask=edge_masks[2])
    elif edge == 'down':
        img_edge_mask = cv2.bitwise_and(image, image, mask=edge_masks[3])
    else:
        print("No valid edge!")

    minV = 30
    maxV = 100
    disp = (img_edge_mask * (255.0 / np.max(img_edge_mask))).astype(np.uint8)
    edges = cv2.Canny(disp,minV,maxV)
    #disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    bla = disp + edges
    cv2.imshow("Canny test",bla)
    cv2.waitKey(0)

def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Numpy file", ".npz")])
    array = np.load(file_path)
    data = array['data']
    extrinsic_params_data = array['extrinsic_params']
    intrinsic_params_data = array['intrinsic_params']
    extrinsic_params = extrinsic_params_data[0, :, :]
    intrinsic_params = intrinsic_params_data[0, :, :]

    is_mask_correct = prepare_images(data, extrinsic_params, intrinsic_params)
    if is_mask_correct == False:
        return

    data_shape = data.shape
    num_frames = data_shape[0]
    mask = target.give_mask(extrinsic_params, intrinsic_params,
                                            shape, center, size, angle)

 
    bias_array = []
    precision_array = []
    for i in range(num_frames):
        depth_image = data[i,:,:,2]
        mean_depth = cv2.mean(depth_image, mask)[0] / 1000

        bias = center[2] - mean_depth
        bias_array.append(bias[0])
    
        diff_matrix = depth_image/1000 - mean_depth
        precision = m.sqrt(np.sum(np.multiply(diff_matrix, diff_matrix)))
        precision_array.append(precision)

        #calculate_edge_precision(depth_image, 'left', extrinsic_params, intrinsic_params)
        
        


    total_bias = sum(bias_array) / len(bias_array)
    total_precision = sum(precision_array) / len(precision_array)

    print("Bias: {:0.3f}, Precision: {:0.3f}".format(total_bias, total_precision))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
