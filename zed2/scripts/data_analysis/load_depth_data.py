import numpy as np
from datetime import datetime
import cv2
import tkinter as tk
from tkinter import filedialog

def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(initialdir="../../logs")

    array = np.load(file_path)
    data = array['data']
    timestamp = array['timestamp']
    date = datetime.fromtimestamp(timestamp[0])

    print(type(data))
    print(data.shape)
    cv2.imshow("ZED | map at {}".format(date), data[0,:,:])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
