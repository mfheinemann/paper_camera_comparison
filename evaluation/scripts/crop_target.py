import cv2
import numpy as np
import math as m


class CropTarget():
    def __init__(self):
        self.mask     = np.array([])
        self.points   = []

    def calculate_3D_points(self, shape, center, size, angle, cam_param):
            
        if shape == 'rectangle':
            rot_z = np.array([[m.cos(angle), -m.sin(angle), 0],
                          [m.sin(angle), m.cos(angle), 0],
                          [0, 0, 1]])

            point_3D = []
            point_3D.append([center[0] - size[0]/2, center[1] - size[1]/2, center[2]])
            point_3D.append([center[0] - size[0]/2, center[1] + size[1]/2, center[2]])
            point_3D.append([center[0] + size[0]/2, center[1] + size[1]/2, center[2]])
            point_3D.append([center[0] + size[0]/2, center[1] - size[1]/2, center[2]])
    
            for point_3D in point_3D:
                point_3D = np.matmul(rot_z, point_3D) 
                point_2D = np.matmul(cam_param, point_3D.append(1))
                point_2D = point_2D / point_2D[2]
                self.points.append(tuple(point_2D[:-1].astype(int)))

        elif shape == 'circle':
            center_2D = np.matmul(cam_param, center.append(1))
            center_2D = center_2D / center_2D[2]
            self.points.append(tuple(center_2D[:-1].astype(int)))

            top_2D = np.matmul(cam_param, center + (0, size, 0))
            center_2D = center_2D / center_2D[2]
            self.points.append(tuple(center_2D[:-1].astype(int)))

        else:
            print("Invalid shape!")
            

    def create_mask(self,image, shape):
        self.mask = np.full((image.shape[0], image.shape[1]), 0, dtype=np.uint8)  # mask is only 
        
        if shape == 'rectangle':
            cv2.fillConvexPoly(self.mask, self.points, (255, 255, 255))
        elif shape == 'circle':
            cv2.circle(self.mask, (self.points[0], self.points[1]), self.points[2], -1)
        else:
            print("Invalid shape!")

    def give_cropped_image(self, image, cam_param, shape, center, size, angle):
        self.calculate_3D_points(shape, center, size, angle, cam_param)
        self.create_mask(image, shape)

        return cv2.bitwise_and(image, image, mask=self.mask)


