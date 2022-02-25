import cv2
import numpy as np
import math as m
import copy

class CropTarget():

    def __init__(self, shape, center, size, angle, edge_width):
        self.shape      = shape
        self.center     = center
        self.size       = size
        self.angle      = angle
        self.edge_width = edge_width

        self.mask          = np.array([])
        self.edge_points   = []
        self.rect_center   = []
        self.rot_points    = []
        self.circle_center = 0
        self.circle_radius = 0
        self._offset_rot_axis = 0.063


    def create_trans_matrix(self, x, z):
        trans_matrix = np.array([[1, x[0], 0],
                                 [0, 1, 0],
                                 [0, z[0], 1]])
        return trans_matrix


    def calculate_edge_points(self):
        '''
            Simplified general rotation around shifted axis

            T_1(x,z)*R_y*T_2(x,z)*(Points) 
        '''

        points_3D = np.array([[self.center[0] - self.size[0]/2, self.center[1] - self.size[1]/2, self.center[2]],
                                [self.center[0] - self.size[0]/2, self.center[1] + self.size[1]/2, self.center[2]],
                                [self.center[0] + self.size[0]/2, self.center[1] + self.size[1]/2, self.center[2]],
                                [self.center[0] + self.size[0]/2, self.center[1] - self.size[1]/2, self.center[2]]])

        rot_y = np.array([[m.cos(self.angle), 0, m.sin(self.angle)],
                          [0, 1, 0],
                          [-m.sin(self.angle), 0, m.cos(self.angle)]])
        translation_matrix_1 = self.create_trans_matrix(
                                self.center[0], self.center[2] + self._offset_rot_axis)
        translation_matrix_2 = self.create_trans_matrix(
                                -self.center[0], -(self.center[2] + self._offset_rot_axis))
        full_rotation = np.matmul(translation_matrix_1, rot_y)
        full_rotation = np.matmul(full_rotation, translation_matrix_2)

        # Set y-value temporarily to 1 for simplified rotation equation
        points_temp = copy.deepcopy(points_3D)
        points_temp[:,1] = 1
        self.rot_points = np.matmul(full_rotation, points_temp)
        self.rot_points = np.squeeze(self.rot_points).transpose()
        self.rot_points[1,:] = points_3D[:,1].transpose()


    def project_shape(self, ex_params, in_params):
        if self.shape == 'rectangle':
            self.calculate_edge_points()
            points_2D, _ = cv2.projectPoints(self.rot_points, ex_params[:,:-1], ex_params[:,3], in_params,0)
            self.edge_points = np.squeeze(points_2D.astype(int))

            center_2D, _ = cv2.projectPoints(self.center, ex_params[:,:-1], ex_params[:,3], in_params,0)
            self.rect_center = np.squeeze(center_2D.astype(int))
        elif self.shape == 'circle':
            center_2D, _ = cv2.projectPoints(self.center, ex_params[:,:-1], ex_params[:,3], in_params,0)
            self.circle_center = tuple(np.squeeze(center_2D.astype(int)))

            top_3D = self.center + np.array([[0], [self.size], [0]])
            top_2D, _ = cv2.projectPoints(top_3D, ex_params[:,:-1], ex_params[:,3], in_params, 0)
            top_2D = tuple(np.squeeze(top_2D.astype(int)))

            radius = m.sqrt((self.circle_center[0] - top_2D[0])*(self.circle_center[0] - top_2D[0]) +
                            (self.circle_center[1] - top_2D[1])*(self.circle_center[1] - top_2D[1]))
            self.circle_radius = int(radius)
        else:
            print("Invalid shape!")


    def create_mask(self, image_dim, increase_edges = False):
        self.mask = np.full((image_dim[0], image_dim[1]), 0, dtype=np.uint8)

        if self.shape == 'rectangle':
            cv2.fillConvexPoly(self.mask, self.edge_points, (255, 255, 255))

            # Increase Polygon edges
            if increase_edges == True:
                cv2.line(self.mask,self.edge_points[0],self.edge_points[1],(255,255,255),self.edge_width)
                cv2.line(self.mask,self.edge_points[1],self.edge_points[2],(255,255,255),self.edge_width)
                cv2.line(self.mask,self.edge_points[2],self.edge_points[3],(255,255,255),self.edge_width)
                cv2.line(self.mask,self.edge_points[3],self.edge_points[0],(255,255,255),self.edge_width)
        elif self.shape == 'circle':
            cv2.circle(self.mask, self.circle_center, self.circle_radius,(255, 255, 255), -1)
        else:
            print("Invalid shape!")


    def give_mask(self, image_dim, ex_params, in_params, increase_edges=False):
        self.project_shape(ex_params, in_params)
        self.create_mask(image_dim)
        mask_out = self.crop_to_target(self.mask, ex_params, in_params, increase_edges)

        return mask_out


    def show_target_in_image(self, image, ex_params, in_params):
        self.project_shape(ex_params, in_params)

        if self.shape == 'rectangle':
            cv2.line(image,self.edge_points[0],self.edge_points[1],(255,0,255),2)
            cv2.line(image,self.edge_points[1],self.edge_points[2],(255,0,255),2)
            cv2.line(image,self.edge_points[2],self.edge_points[3],(255,0,255),2)
            cv2.line(image,self.edge_points[3],self.edge_points[0],(255,0,255),2)
        elif self.shape == 'circle':
            cv2.circle(image, self.circle_center, self.circle_radius,(255, 0, 255), 2)
        else:
            print("Invalid shape!")

        return image

    def crop_to_target(self, image, ex_params, in_params, increase_edges=False):
        self.project_shape(ex_params, in_params)

        if self.shape == 'rectangle':
            max_x = np.max(self.edge_points[:,0])
            min_x = np.min(self.edge_points[:,0])
            max_y = np.max(self.edge_points[:,1])
            min_y = np.min(self.edge_points[:,1])
        elif self.shape == 'circle':
            max_x = self.circle_center[0] + self.circle_radius
            min_x = self.circle_center[0] - self.circle_radius
            max_y = self.circle_center[1] + self.circle_radius
            min_y = self.circle_center[1] - self.circle_radius
        else:
            print("Invalid shape!")

        # Add dimension to enable subsequent cropping of image or point cloud
        img_dim = image.shape
        if len(img_dim) == 2:
            image = image.reshape(img_dim[0], img_dim[1],1)

        if increase_edges:
            image_out = image[min_y-self.edge_width : max_y+self.edge_width, 
                            min_x-self.edge_width : max_x+self.edge_width, :]
        else:
            image_out = image[min_y:max_y, min_x:max_x, :]

        return image_out


    def create_edge_masks(self, image_dim, ex_params, in_params):
        mask_edge_left = np.full((image_dim[0], image_dim[1]), 0, dtype=np.uint8)
        mask_edge_down = np.full((image_dim[0], image_dim[1]), 0, dtype=np.uint8)
        mask_edge_right = np.full((image_dim[0], image_dim[1]), 0, dtype=np.uint8)
        mask_edge_up = np.full((image_dim[0], image_dim[1]), 0, dtype=np.uint8)

        offset_parameter = int(0.1* m.sqrt((self.edge_points[1,0] - self.edge_points[0,0])**2 +
                                  (self.edge_points[1,1] - self.edge_points[0,1])**2))

        move_leftright = np.array([offset_parameter, 0])
        move_updown = np.array([0, offset_parameter])

        self.project_shape(ex_params, in_params)
        if self.shape == 'rectangle':
            # LEFT
            points = np.vstack((self.edge_points[(0,1),:], self.rect_center))
            cv2.fillConvexPoly(mask_edge_left, points, 255)
            cv2.line(mask_edge_left, self.edge_points[0]+move_updown, self.edge_points[1]-move_updown,
                    (255, 255, 255), self.edge_width)

            # DOWN
            points = np.vstack((self.edge_points[(1,2),:], self.rect_center))
            cv2.fillConvexPoly(mask_edge_down, points, 255)
            cv2.line(mask_edge_down, self.edge_points[1]+move_leftright, self.edge_points[2]-move_leftright,
                    (255, 255, 255), self.edge_width)

            # RIGHT
            points = np.vstack((self.edge_points[(2,3),:], self.rect_center))
            cv2.fillConvexPoly(mask_edge_right, points, 255)
            cv2.line(mask_edge_right, self.edge_points[2]-move_updown, self.edge_points[3]+move_updown,
                    (255, 255, 255), self.edge_width)

            # DOWN
            points = np.vstack((self.edge_points[(3,0),:], self.rect_center))
            cv2.fillConvexPoly(mask_edge_up, points, 255)
            cv2.line(mask_edge_up, self.edge_points[3]-move_leftright, self.edge_points[0]+move_leftright,
                    (255, 255, 255), self.edge_width)
        else:
            print("Invalid shape!")

        return (mask_edge_left, mask_edge_down, mask_edge_right, mask_edge_up)
