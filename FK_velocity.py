import numpy as np 
from lib.calcJacobian import calcJacobian

def FK_velocity(q_in, dq):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param dq: 1 x 7 vector corresponding to the joint velocities.
    :return:
    velocity - 6 x 1 vector corresponding to the end effector velocities.    
    """

    ## STUDENT CODE GOES HERE
    J = calcJacobian(q_in)
    velocity = J @ np.transpose(dq)
    return velocity
   
if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    qDot = np.array([1,0,0,0,0,0,0])
    print(FK_velocity(q, qDot))
