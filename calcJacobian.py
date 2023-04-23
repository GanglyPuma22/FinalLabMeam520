import numpy as np
from calculateFK import FK
    
def calcJacobian(q_in, jointIndex=7):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))

    ## STUDENT CODE GOES HERE
    fk = FK()
    jointPositions, T0e = fk.forward(q_in)
    rotations = fk.compute_Ai(q_in)
    #Z = fk.get_axis_of_rotation(q_in)
    Jv = np.zeros((3, 7))
    Jw = np.zeros((3, 7))
    currentRotation = np.identity(4)
    
    for i in range(1, 8):
    	linkDistance = jointPositions[7,:] - jointPositions[i-1,:]
    	Jv[:,i-1] = np.cross(rotations[:, :, i-1][:3,2], linkDistance)
    	Jw[:,i-1] = rotations[:, :, i-1][:3,2]
    
    J = np.concatenate((Jv, Jw))
    return J

if __name__ == '__main__':
    #q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    q = np.array([0,0,0,0,0,0,0])
    print(np.round(calcJacobian(q),3))
