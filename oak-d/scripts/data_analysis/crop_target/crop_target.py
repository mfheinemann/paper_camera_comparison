import cv2
import numpy as np
import math as m
import copy

class CropTarget():

    def __init__(self):
        self.mask        = np.array([])
        self.edge_points = []
        self.rot_points  = []
        self.center      = 0
        self.radius      = 0

    def create_trans_matrix(self, x, z):
        trans_matrix = np.array([[1, x[0], 0],
                                 [0, 1, 0],
                                 [0, z[0], 1]])
        return trans_matrix

    
    def calculate_edge_points(self, center, size, angle):
        '''
            Simplified general rotation around shifted axis
        
            T_1(x,z)*R_y*T_2(x,z)*(Points) 
        '''

        points_3D = np.array([[center[0] - size[0]/2, center[1] - size[1]/2, center[2]],
                                [center[0] - size[0]/2, center[1] + size[1]/2, center[2]],
                                [center[0] + size[0]/2, center[1] + size[1]/2, center[2]],
                                [center[0] + size[0]/2, center[1] - size[1]/2, center[2]]])

        rot_y = np.array([[m.cos(angle), 0, m.sin(angle)],
                          [0, 1, 0],
                          [-m.sin(angle), 0, m.cos(angle)]])
        translation_matrix_1 = self.create_trans_matrix(
                                center[0], center[2])
        translation_matrix_2 = self.create_trans_matrix(
                                -center[0], -center[2])
        full_rotation = np.matmul(translation_matrix_1, rot_y)
        full_rotation = np.matmul(full_rotation, translation_matrix_2)

        # Set y-value temporarily to 1 for simplified rotation equation
        points_temp = copy.deepcopy(points_3D)
        points_temp[:,1] = 1
        self.rot_points = np.matmul(full_rotation, points_temp)
        self.rot_points = np.squeeze(self.rot_points).transpose()
        self.rot_points[1,:] = points_3D[:,1].transpose()


    def project_shape(self, shape, center, size, angle, ex_params, in_params):
        if shape == 'rectangle':
            self.calculate_edge_points(center, size, angle)
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
            

    def create_mask(self, shape):
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


    def show_target_in_image(self, image, ex_params, in_params, shape, center, size, angle):
        self.reset_values(image)
        self.project_shape(shape, center, size, angle, ex_params, in_params)

        if shape == 'rectangle':
            print(self.edge_points[0])
            print(self.edge_points[1])
            cv2.line(image,self.edge_points[0],self.edge_points[1],(255,0,0),2)
            cv2.line(image,self.edge_points[1],self.edge_points[2],(255,0,0),2)
            cv2.line(image,self.edge_points[2],self.edge_points[3],(255,0,0),2)
            cv2.line(image,self.edge_points[3],self.edge_points[0],(255,0,0),2)
        elif shape == 'circle':
            cv2.circle(image, self.center, self.radius,(0, 0, 255), 2)
        else:
            print("Invalid shape!")
        
        return image
