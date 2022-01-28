import cv2
import numpy as np
import math as m
import copy

class CropTarget():

    def __init__(self):
        self.mask            = np.array([])
        self.mask_edge_left  = np.array([])
        self.mask_edge_right = np.array([])
        self.mask_edge_up    = np.array([])
        self.mask_edge_down  = np.array([])

        self.edge_points   = []
        self.rot_points    = []
        self.circle_center = 0
        self.circle_radius = 0
        self._offset_rot_axis = 0.063


    def create_trans_matrix(self, x, z):
        trans_matrix = np.array([[1, x[0], 0],
                                 [0, 1, 0],
                                 [0, z[0], 1]])
        return trans_matrix

    
    def calculate_edge_points(self, center_in, size_in, angle_in):
        '''
            Simplified general rotation around shifted axis
        
            T_1(x,z)*R_y*T_2(x,z)*(Points) 
        '''

        points_3D = np.array([[center_in[0] - size_in[0]/2, center_in[1] - size_in[1]/2, center_in[2]],
                                [center_in[0] - size_in[0]/2, center_in[1] + size_in[1]/2, center_in[2]],
                                [center_in[0] + size_in[0]/2, center_in[1] + size_in[1]/2, center_in[2]],
                                [center_in[0] + size_in[0]/2, center_in[1] - size_in[1]/2, center_in[2]]])

        rot_y = np.array([[m.cos(angle_in), 0, m.sin(angle_in)],
                          [0, 1, 0],
                          [-m.sin(angle_in), 0, m.cos(angle_in)]])
        translation_matrix_1 = self.create_trans_matrix(
                                center_in[0], center_in[2] + self._offset_rot_axis)
        translation_matrix_2 = self.create_trans_matrix(
                                -center_in[0], -(center_in[2] + self._offset_rot_axis))
        full_rotation = np.matmul(translation_matrix_1, rot_y)
        full_rotation = np.matmul(full_rotation, translation_matrix_2)

        # Set y-value temporarily to 1 for simplified rotation equation
        points_temp = copy.deepcopy(points_3D)
        points_temp[:,1] = 1
        self.rot_points = np.matmul(full_rotation, points_temp)
        self.rot_points = np.squeeze(self.rot_points).transpose()
        self.rot_points[1,:] = points_3D[:,1].transpose()


    def project_shape(self, shape_in, center_in, size_in, angle_in, ex_params, in_params):
        if shape_in == 'rectangle':
            self.calculate_edge_points(center_in, size_in, angle_in)
            points_2D, jac = cv2.projectPoints(self.rot_points, ex_params[:,:-1], ex_params[:,3], in_params,0)
            self.edge_points = np.squeeze(points_2D.astype(int))

        elif shape_in == 'circle':
            center_2D, jac = cv2.projectPoints(center_in, ex_params[:,:-1], ex_params[:,3], in_params,0)
            self.circle_center = tuple(np.squeeze(center_2D.astype(int)))

            top_3D = center_in + np.array([[0], [size_in], [0]])
            top_2D, jac = cv2.projectPoints(top_3D, ex_params[:,:-1], ex_params[:,3], in_params, 0)
            top_2D = tuple(np.squeeze(top_2D.astype(int)))

            radius = m.sqrt((self.circle_center[0] - top_2D[0])*(self.circle_center[0] - top_2D[0]) +
                            (self.circle_center[1] - top_2D[1])*(self.circle_center[1] - top_2D[1]))
            self.circle_radius = int(radius)
        else:
            print("Invalid shape!")
            

    def create_mask(self, shape_in):
        if shape_in == 'rectangle':
            cv2.fillConvexPoly(self.mask, self.edge_points, (255, 255, 255))
        elif shape_in == 'circle':
            cv2.circle(self.mask, self.circle_center, self.circle_radius,(255, 255, 255), -1)
        else:
            print("Invalid shape!")


    def give_mask(self, ex_params, in_params, shape_in, center_in, size_in, angle_in):
        self.project_shape(shape_in, center_in, size_in, angle_in, ex_params, in_params)
        self.create_mask(shape_in)

        return self.mask


    def give_cropped_image(self, image, ex_params, in_params, shape_in, center_in, size_in, angle_in):
        self.reset_values(image)
        self.project_shape(shape_in, center_in, size_in, angle_in, ex_params, in_params)
        self.create_mask(shape_in)

        return cv2.bitwise_and(image, image, mask=self.mask)


    def create_edge_masks(self, image, ex_params, in_params, shape_in, center_in, size_in,
                                    angle_in, edge_width):
        self.reset_values(image)
        self.project_shape(shape_in, center_in, size_in, angle_in, ex_params, in_params)

        if shape_in == 'rectangle':
            cv2.line(self.mask_edge_left, self.edge_points[0], self.edge_points[1],
                    (255, 255, 255), edge_width)
            cv2.line(self.mask_edge_up, self.edge_points[1], self.edge_points[2],
                    (255, 255, 255), edge_width)
            cv2.line(self.mask_edge_right, self.edge_points[2], self.edge_points[3],
                    (255, 255, 255), edge_width)
            cv2.line(self.mask_edge_down, self.edge_points[3], self.edge_points[0],
                    (255, 255, 255), edge_width)
        elif shape_in == 'circle':
            cv2.circle(self.mask, self.circle_center, self.circle_radius,
                    (255, 255, 255), edge_width)
        else:
            print("Invalid shape!")
        
        return (self.mask_edge_left, self.mask_edge_up, self.mask_edge_right, self.mask_edge_down)


    def show_target_in_image(self, image, ex_params, in_params, shape_in, center_in, size_in, angle_in):
        self.reset_values(image)
        self.project_shape(shape_in, center_in, size_in, angle_in, ex_params, in_params)

        if shape_in == 'rectangle':
            cv2.line(image,self.edge_points[0],self.edge_points[1],(255,0,255),2)
            cv2.line(image,self.edge_points[1],self.edge_points[2],(255,0,255),2)
            cv2.line(image,self.edge_points[2],self.edge_points[3],(255,0,255),2)
            cv2.line(image,self.edge_points[3],self.edge_points[0],(255,0,255),2)
        elif shape_in == 'circle':
            cv2.circle(image, self.circle_center, self.circle_radius,(255, 0, 255), 2)
        else:
            print("Invalid shape!")
        
        return image


    def reset_values(self, image):
        self.edge_points = []
        self.rot_points = []
        self.circle_center = 0
        self.circle_radius = 0
        self.mask = np.full((image.shape[0], image.shape[1]), 0, dtype=np.uint8)
        self.mask_edge_left  = np.full((image.shape[0], image.shape[1]), 0, dtype=np.uint8)
        self.mask_edge_right = np.full((image.shape[0], image.shape[1]), 0, dtype=np.uint8)
        self.mask_edge_up    = np.full((image.shape[0], image.shape[1]), 0, dtype=np.uint8)
        self.mask_edge_down  = np.full((image.shape[0], image.shape[1]), 0, dtype=np.uint8)
