import numpy as np
from datetime import datetime
import cv2
import pyransac3d as pyrsc
import tkinter as tk
from tkinter import filedialog

def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(initialdir="../../logs")

    array = np.load(file_path, allow_pickle=True)
    data = array['data']
    timestamp = array['timestamp']

    test = data[1]
    print(test.shape)
    print(test[1])

    plane = pyrsc.Plane()
    equation, inliers = plane.fit(data[1], thresh=0.05, minPoints=100, maxIteration=1000)
    print(equation)
    print(inliers)


if __name__ == "__main__":
    main()
