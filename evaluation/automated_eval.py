import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import os
import setup_1_bias
import setup_2_adr
import setup_3_radius
import setup_3_sphere
from common.constants import *
from common.experiments import *
from crop_target.crop_target import CropTarget

def main():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()

    for path, dirs, files in os.walk(folder_path):
        for name in files:
            shape   = 'rectangle'
            center  = np.array([[0.0], [0.0], [1.0 - OFFSET['zed2']]])
            size    = np.asarray(TARGET_SIZE) - REDUCE_TARGET
            angle   = 0.0
            edge_width = EDGE_WIDTH

            setup =0

            if name[-4:] == '.npz':
                camera, number = name[4:-7].split('_')
                if int(number) in SETUPS['1']['experiments']: setup = 1
                if int(number) in SETUPS['2']['experiments']: setup = 2
                if int(number) in SETUPS['3']['experiments']: setup = 3

                if setup == 1:
                    ind = SETUPS['1']['experiments'].index(int(number))
                    center[2] = SETUPS['1']['distances'][ind] - OFFSET[camera]

                    target  = CropTarget(shape, center, size, angle, edge_width)

                    setup_1_bias.eval_setup_1(os.path.join(path, name), target,
                        shape, center, size, angle, edge_width)

                if setup == 2:
                    ind = SETUPS['2']['experiments'].index(int(number))
                    center[2] = 2.0 - OFFSET[camera]
                    angle = -SETUPS['2']['angles'][ind]
                    if camera == 'orbbec': angle = -angle

                    target  = CropTarget(shape, center, size, angle, edge_width)

                    setup_2_adr.eval_setup_2(os.path.join(path, name), target,
                        shape, center, size, angle, edge_width)

                if setup == 3:
                    ind = SETUPS['3']['experiments'].index(int(number))
                    center[2] = SETUPS['3']['distances'][ind] - OFFSET[camera]
                    shape = 'circle'

                    target  = CropTarget(shape, center, size, angle, edge_width)

                    setup_3_radius.eval_setup_3_1(os.path.join(path, name), target,
                        shape, center, size, angle, edge_width)

                    setup_3_sphere.eval_setup_3_2(os.path.join(path, name), target,
                        shape, center, size, angle, edge_width)


    return

if __name__ == "__main__":
    main()
