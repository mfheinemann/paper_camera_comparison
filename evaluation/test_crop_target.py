from crop_target.crop_target import CropTarget
import numpy as np
import tkinter as tk
from tkinter import filedialog
import cv2

def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(initialdir="../../logs")

    array = np.load(file_path)
    data = array['data']
    extrinsic_params_data = array['extrinsic_params']
    intrinsic_params_data = array['intrinsic_params']
    image = data[0,:,:,2]
    extrinsic_params = extrinsic_params_data[0, :, :]
    intrinsic_params = intrinsic_params_data[0, :, :]

    # Define target
    shape   = 'circle'
    if shape == 'rectangle':
        center  = np.array([[1.0], [0.0], [3.0]])    # Center of plane
        size    = np.array([0.2, 0.2])               # (width, height) in m
        angle   = 0.0                                # In degrees
    elif shape == 'circle':
        center  = np.array([[1.0], [0.0], [3.0]])   # Center of shpere
        size    = 0.2                               # Radius in m
        angle   = 0.0
    else:
        print("Not a valid shape!")

    target = CropTarget()
    disp = (image * (255.0 / np.max(image))).astype(np.uint8)
    disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    image_new = target.give_cropped_image(disp, extrinsic_params, intrinsic_params,
                                            shape, center, size, angle)
    image_new2 = target.show_target_in_image(disp, extrinsic_params, intrinsic_params,
                                            shape, center, size, angle)

    cv2.imshow("cropped image", image_new)
    cv2.imshow("cropped image2", image_new2)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
