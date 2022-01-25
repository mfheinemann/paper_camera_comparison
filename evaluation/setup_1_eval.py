# Michel Heinemann
# calulate bias, precision and edge precision from depth data

import numpy as np
from datetime import datetime
import cv2
import tkinter as tk
from tkinter import filedialog
from crop_target.crop_target import CropTarget


# Define target
target = CropTarget()
shape   = 'rectangle'
if shape == 'rectangle':
    center  = np.array([[0.0], [0.0], [2.983]])    # Center of plane
    size    = np.array([0.48, 0.48])               # (width, height) in m
    angle   = 0.0                                # In degrees
elif shape == 'circle':
    center  = np.array([[1.0], [0.0], [3.0]])   # Center of shpere
    size    = 0.2                               # Radius in m
    angle   = 0.0
else:
    print("Not a valid shape!")

def prepare_images(data, extrinsic_params, intrinsic_params):
    depth_image = data[0,:,:,2]
    disp = (depth_image * (255.0 / np.max(depth_image))).astype(np.uint8)
    disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    image_mask = target.give_cropped_image(disp, extrinsic_params, intrinsic_params,
                                            shape, center, size, angle)
    image_frame = target.show_target_in_image(disp, extrinsic_params, intrinsic_params,
                                            shape, center, size, angle)
    img_stacked = np.hstack((image_mask, image_frame))
    cv2.imshow("If mask is applied correctly, press 'q'", img_stacked)
    key = cv2.waitKey(0)
    if key == 113:
        print("Mask applied correctly")
        cv2.destroyAllWindows()
        return True
    else:
        print("Mask applied incorrectly, please adjust target parameters...")
        return False


def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
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

 
    key = cv2.waitKey(0)
    mean_array = []
    for i in range(num_frames):
        depth_image = data[i,:,:,2]
        mean_depth = cv2.mean(depth_image, mask)[0] / 1000
        print(mean_depth)
        mean_array.append(mean_depth)

    total_mean = sum(mean_array) / len(mean_array)
    print("Total Mean: ", total_mean)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
