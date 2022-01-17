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
    img = data[0,:,:]
    
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    wide = cv2.Canny(blurred, 10, 200)
    tight = cv2.Canny(blurred, 225, 250)

    cv2.imshow("Original", img)
    cv2.imshow("Blur",blurred)
    cv2.imshow("Edges", np.hstack([wide, tight]))
    cv2.waitKey(0)

    minV = 30
    maxV = 100

    edges = cv2.Canny(img,minV,maxV)
    kernel = np.ones((7,7),np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    

   # cv2.imshow('Original',img)
    #cv2.imshow('Canny',edges)
    #cv2.imshow('gradient',gradient)

    #cv2.imshow("ZED | map at {}".format(date), data[0,:,:])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
