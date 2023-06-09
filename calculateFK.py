
import numpy as np
from math import pi

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout
        
        
        pass
    
    def createRotationMatrix(self, q, alpha, a, d):
            return np.matrix([[np.cos(q), -np.sin(q)*np.cos(alpha), np.sin(alpha)*np.sin(q), a*np.cos(q)],
             [np.sin(q), np.cos(q)*np.cos(alpha), -np.cos(q)*np.sin(alpha), a*np.sin(q)],
              [0, np.sin(alpha), np.cos(alpha), d],
               [0, 0, 0, 1]])
        
    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here

        jointPositions = np.zeros((8,3))
        T0e = np.identity(4)
        
        #rotations = [R0_1, R1_2, R2_3, R3_4, R4_5, R5_6, R6_e]
        rotations = np.zeros((4,4,8))
        rotations[:,:,0] = np.identity(4)
        
        frameOffsets = np.matrix([ 
            [0, 0, 0.141, 1],
            [0, 0, 0, 1],
            [0, 0, 0.195, 1],
            [0, 0, 0, 1],
            [0, 0, 0.125, 1],
            [0, 0, -0.015, 1],
            [0, 0, 0.051, 1],
            [0, 0, 0, 1]
        ]
        )
        
        rotations = self.compute_Ai(q)
        
        for i in range(0, 7):
            finalPos = rotations[:,:,i+1] @ np.transpose(frameOffsets[i+1])
            jointPositions[i+1] = np.resize(finalPos, 3)
            
        T0e = rotations[:,:,7]

        #Update first joint position too because for loop starts at index 1   
        jointPositions[0] = jointPositions[0] + np.resize(frameOffsets[0], 3)   
                
        
        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    
    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2
        rotations = self.compute_Ai(q)
        Z = np.zeros((3,7))
        for i in range(0,7):
            Z[:,i-1] = rotations[:,:,i][:3,2]
            
        return Z
    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2
        l1 = 0.333
        l2 = 0.195 + 0.121
        l4 = 0.0825
        l5 = 0.125+0.259 #= 0.384
        l6=0.088
        l7=0.210
        
        angles = [q[0], q[1], q[2], q[3], q[4], q[5], q[6] - pi/4]
        alph = [-pi/2, pi/2, pi/2, -pi/2, pi/2, pi/2, 0]
        a = [0, 0, l4, -l4, 0, l6, 0]
        d = [l1, 0, l2, 0, l5, 0, l7]
        
        rotations = np.zeros((4,4,8))
        rotations[:,:,0] = np.identity(4)
        
        for i in range(0, 7):
            rotationMatrix = self.createRotationMatrix(angles[i], alph[i], a[i], d[i])
            rotations[:,:,i+1] = rotations[:,:,i] @ rotationMatrix

        return rotations
    
if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    joint_positions, T0e = fk.forward(q)
    z = fk.get_axis_of_rotation(q)
    print(np.shape(z))
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
