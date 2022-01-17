import sys
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir="../../logs")

array = np.load(file_path)
im = array['0.9.npy']
imgray = im[:,:]
#imgray = cv2.cvtColor(imgray, cv2.COLOR_BGR2GRAY)
imgray = imgray.astype(np.uint8)

#imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow('image',imgray)
cv2.waitKey(0)
# Thresholding - delete all points outside 60cm - 150cm
ret, thresh = cv2.threshold(imgray, 10 , 240, 0)
cv2.imshow('thresh',thresh)
cv2.waitKey(0)

kernel = np.ones((7,7),np.uint8)
erosion = cv2.erode(thresh, kernel, iterations = 4)
dilation = cv2.dilate(erosion, kernel, iterations = 4)
cv2.imshow('dilation', dilation)
cv2.waitKey(0)


contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(contours)
cv2.drawContours(imgray, contours, -1, (255,255,0), 3)
cv2.imshow('contours',imgray)
cv2.waitKey(0)

(y,x),radius = cv2.minEnclosingCircle(contours[8])
print(x)
print(y)
center = (int(x),int(y))
px = imgray[int(x), int(y)]
print("Center Position in pixel: {} distance_value: {}".format(center, px))
radius = int(radius)
cv2.circle(imgray,center,radius,(255,100,100),5)
cv2.circle(imgray,center,1,(100,100,255),5)
cv2.imshow('result',imgray)
cv2.waitKey(0)
cv2.destroyAllWindows() 
