import sys
import numpy as np
import cv2


array = np.load("../../logs/log_zed2_220114131813.npz")
im = array['data']
print(im.shape)
imgray = im[0,:,:]
imgray *= 100
imgray = imgray.astype('uint8')

#imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow('image',imgray)
cv2.waitKey(0)
# Thresholding - delete all points outside 60cm - 150cm
ret, thresh = cv2.threshold(imgray, 70 , 100, 0)
cv2.imshow('thresh',thresh)
cv2.waitKey(0)

kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(thresh, kernel, iterations = 4)
dilation = cv2.dilate(erosion, kernel, iterations = 4)
cv2.imshow('dilation', dilation)
cv2.waitKey(0)


contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
cv2.drawContours(imgray, contours, -1, (255,255,0), 3)
cv2.imshow('contours',imgray)
cv2.waitKey(0)

(x,y),radius = cv2.minEnclosingCircle(contours[8])
print(x)
print(y)
center = (int(x),int(y))
px = imgray[int(y), int(x)]
print("Center Position in pixel: {} distance_value: {}".format(center, px))
radius = int(radius)
cv2.circle(imgray,center,radius,(255,0,0),2)
cv2.circle(imgray,center,1,(255,0,255),2)
cv2.imshow('result',imgray)
cv2.waitKey(0)
cv2.destroyAllWindows() 
