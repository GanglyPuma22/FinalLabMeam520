import numpy as np 
from lib.calcJacobian import calcJacobian



def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    ## STUDENT CODE GOES HERE
    desiredV = np.transpose(np.hstack((v_in, omega_in)))
    J = calcJacobian(q_in)
    desiredV = np.concatenate((v_in, omega_in))
    
    #Unconstrain nan velocities
    nanVel = ~np.isnan(desiredV)
    desiredV = desiredV[nanVel]
    J = J[nanVel]
    #Find least squared solution
    normSol = np.linalg.lstsq(J, desiredV)

    return np.round(np.array(normSol[0]), 3)

if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    #q = np.array([0,0,0,0,0,0,0])
    print(IK_velocity(q, np.array([0, 1, 0]), np.array([np.nan, np.nan, np.nan])))
