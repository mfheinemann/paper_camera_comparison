# Michel Heinemann
# calulate bias, precision and edge precision from depth data

import numpy as np
from datetime import datetime
import cv2
import tkinter as tk
from tkinter import filedialog

def main():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()

    array = np.load(file_path)
    for key, array in array.items():
        print(array)
        img_16 = cv2.cvtColor(array,cv2.COLOR_GRAY2BGR)
        img_8 = cv2.convertScaleAbs(img_16)
        edges = cv2.Canny(image=img_8, threshold1=1, threshold2=100)

        # Display Canny Edge Detection Image
        cv2.imshow('edges', edges)
        cv2.waitKey(1)


    # data = array.files
    # print(array)
    # data = array['data']
    # timestamp = array['timestamp']

if __name__ == "__main__":
    main()