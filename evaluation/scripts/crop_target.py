import cv2
import numpy as np
import math as m


class CropTarget():
    def __init__(self):
        self.mask        = np.array([])
        self.edge_points = []
        self.rot_points  = []
        self.center      = 0
        self.radius      = 0

    def calculate_edge_points(self, center, size, angle):
        rot_z = np.array([[m.cos(angle), -m.sin(angle), 0],
                              [m.sin(angle), m.cos(angle), 0],
                              [0, 0, 1]])

        points_3D = np.array([[center[0] - size[0]/2, center[1] - size[1]/2, center[2]],
                                [center[0] - size[0]/2, center[1] + size[1]/2, center[2]],
                                [center[0] + size[0]/2, center[1] + size[1]/2, center[2]],
                                [center[0] + size[0]/2, center[1] - size[1]/2, center[2]]])

        self.rot_points = [np.matmul(rot_z, point) for point in points_3D]

    def project_shape(self, shape, center, size, angle, ex_params, in_params):

        if shape == 'rectangle':
            self.calculate_edge_points()
            points_2D, jac = cv2.projectPoints(self.rot_points, ex_params[:,:-1], ex_params[:,3], in_params,0)
            self.edge_points = np.squeeze(points_2D.astype(int))

        elif shape == 'circle':
            center_2D, jac = cv2.projectPoints(center, ex_params[:,:-1], ex_params[:,3], in_params,0)
            self.center = tuple(np.squeeze(center_2D.astype(int)))

            top_3D = center + np.array([[0], [size], [0]])
            top_2D, jac = cv2.projectPoints(top_3D, ex_params[:,:-1], ex_params[:,3], in_params, 0)
            top_2D = tuple(np.squeeze(top_2D.astype(int)))

            radius = m.sqrt((self.center[0] - top_2D[0])*(self.center[0] - top_2D[0]) +
                            (self.center[1] - top_2D[1])*(self.center[1] - top_2D[1]))
            self.radius = int(radius)
        else:
            print("Invalid shape!")
            

    def create_mask(self,shape):
        
        if shape == 'rectangle':
            cv2.fillConvexPoly(self.mask, self.edge_points, (255, 255, 255))
        elif shape == 'circle':
            cv2.circle(self.mask, self.center, self.radius,(255, 255, 255), -1)
        else:
            print("Invalid shape!")


    def reset_values(self, image):
        self.edge_points = []
        self.rot_points  = []
        self.center = 0
        self.radius = 0
        self.mask   = np.full((image.shape[0], image.shape[1]), 0, dtype=np.uint8)


    def give_cropped_image(self, image, ex_params, in_params, shape, center, size, angle):
        self.reset_values(image)
        self.project_shape(shape, center, size, angle, ex_params, in_params)
        self.create_mask(shape)

        return cv2.bitwise_and(image, image, mask=self.mask)
