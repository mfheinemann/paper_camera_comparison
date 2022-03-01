import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import os
import setup_1_bias
import setup_1_edge
import setup_2_adr
import setup_3_radius
import setup_3_sphere
from common.constants import *
from common.experiments import *
from crop_target.crop_target import CropTarget
import csv
from pathlib import Path

def main():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()

    result_path = os.path.join(folder_path, 'results.csv')
    init_row = ['camera', 'number', 'setup', 'result_1', 'result_2', 'result_3', 'result_4', 'result_5', 'result_6']
    append_list_as_row(result_path, init_row)

    images_path = os.path.join(folder_path, 'result_images')
    Path(images_path).mkdir(parents=True, exist_ok=True)

    for path, dirs, files in os.walk(folder_path):
        for name in files:
            shape   = 'rectangle'
            center  = np.array([[0.0], [0.0], [3.0 - OFFSET['zed2']]])
            size    = np.asarray(TARGET_SIZE) - REDUCE_TARGET
            angle   = 0.0
            edge_width = EDGE_WIDTH

            setup = 0

            if name[-4:] == '.npz':
                camera, number = name[4:-7].split('_')
                if int(number) in SETUPS['1']['experiments']: setup = 1
                if int(number) in SETUPS['2']['experiments']: setup = 2
                if int(number) in SETUPS['3']['experiments']: setup = 3

                if setup == 1:
                    ind = SETUPS['1']['experiments'].index(int(number))
                    center[2] = SETUPS['1']['distances'][ind] - OFFSET[camera]

                    target  = CropTarget(shape, center, size, angle, edge_width)

                    bias, precision, nan_ratio, edge_precision, nan_edge_ratio, \
                        first_image_with_target = setup_1_bias.eval_setup_1(
                            os.path.join(path, name), target, center, size, angle,
                            edge_width, False)

                    results = [camera, number, setup, bias, precision, nan_ratio, edge_precision, nan_edge_ratio]

                elif setup == 2:
                    ind = SETUPS['2']['experiments'].index(int(number))
                    center[2] = 2.0 - OFFSET[camera]
                    angle = np.radians(-SETUPS['2']['angles'][ind])

                    if camera == 'orbbec': angle = -angle

                    target  = CropTarget(shape, center, size, angle, edge_width)

                    adr, std, first_image_with_target = setup_2_adr.eval_setup_2(
                        os.path.join(path, name), target, center, size, angle,
                        edge_width, False)

                    results = [camera, number, setup, std]

                elif setup == 3:
                    ind = SETUPS['3']['experiments'].index(int(number))
                    center[2] = SETUPS['3']['distances'][ind] - OFFSET[camera]
                    shape = 'circle'
                    size = SPHERE_RADIUS

                    target  = CropTarget(shape, center, size, angle, edge_width)

                    radius_mean, radius_std, first_image_with_target = setup_3_radius.eval_setup_3_1(
                        os.path.join(path, name), target, center, size, angle, edge_width, False)

                    sphere_rec_error, sphere_pos, _ = setup_3_sphere.eval_setup_3_2(
                        os.path.join(path, name), target, center, size, angle, edge_width, False)

                    results = [camera, number, setup, radius_mean, radius_std, sphere_rec_error, sphere_pos]

                else: continue

                append_list_as_row(result_path, results)
                
                image_name = camera + '_' + str(number) + '.png' 
                cv2.imwrite(os.path.join(images_path, image_name), first_image_with_target)


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

if __name__ == "__main__":
    main()
