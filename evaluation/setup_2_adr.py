import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from crop_target.crop_target import CropTarget
from common.constants import *

# Define target
shape   = 'rectangle'
center  = np.array([[0.0], [0.0], [2.0 - OFFSET['oak']]])
size    = np.asarray(TARGET_SIZE) - REDUCE_TARGET
angle   = np.radians(-20.0)
edge_width = 0
target  = CropTarget(shape, center, size, angle, edge_width)


def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Numpy file", ".npz")])

    print("Opening file: ", file_path, "\n")
    print("Experiment configuration - Setup 2 (ADR)\nDistance:\t{:.3f}m\nTarget size:\t({:.3f},{:.3f})m\nAngle:\t\t{:.3f}rad\nEdge width:\t{}px".format(
         np.squeeze(center[2]), np.squeeze(size[0]), np.squeeze(size[1]), angle, edge_width))

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

    valid_points_ratio = np.zeros((num_frames, 1))
    for i in range(num_frames):
        depth_image = data[i,:,:,2].astype(np.int16)
        image_cropped = target.crop_to_target(depth_image, extrinsic_params, intrinsic_params)

        not_nan = ~np.isnan(image_cropped[mask_bool])
        not_zero = image_cropped[mask_bool] > 0

        valid_points_ratio[i] = (not_nan & not_zero).sum() / mask_bool.sum()

    total_adr = np.mean(valid_points_ratio)
    print("Angle-dependend Reflectivity: {:0.3f}".format(total_adr))

    cv2.destroyAllWindows()


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


if __name__ == "__main__":
    main()
