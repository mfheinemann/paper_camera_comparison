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
    data = array.files
    print(data)
    # data = array['data']
    # timestamp = array['timestamp']

if __name__ == "__main__":
    main()