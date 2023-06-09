import numpy as np
from math import pi, acos, sin, cos
from scipy.linalg import null_space
from numpy import linalg

from calcJacobian import calcJacobian
from calculateFK import FK
from IK_velocity import IK_velocity


class IK:
    # JOINT LIMITS
    lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    center = lower + (upper - lower) / 2  # compute middle of range of motion of each joint
    fk = FK()

    def __init__(self, linear_tol=1e-3, angular_tol=5e-3, max_steps=10000, min_step_size=1e-5):
        """
        Constructs an optimization-based IK solver with given solver parameters.
        Default parameters are tuned to reasonable values.

        PARAMETERS:
        linear_tol - the maximum distance in meters between the target end
        effector origin and actual end effector origin for a solution to be
        considered successful
        angular_tol - the maximum angle of rotation in radians between the target
        end effector frame and actual end effector frame for a solution to be
        considered successful
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # solver parameters
        self.linear_tol = linear_tol
        self.angular_tol = angular_tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size

    ######################
    ## Helper Functions ##
    ######################

    @staticmethod
    def displacement_and_axis(target, current):
        """
        Helper function for the End Effector Task. Computes the displacement
        vector and axis of rotation from the current frame to the target frame

        This data can also be interpreted as an end effector velocity which will
        bring the end effector closer to the target position and orientation.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        current - 4x4 numpy array representing the "current" end effector orientation

        OUTPUTS:
        displacement - a 3-element numpy array containing the displacement from
        the current frame to the target frame, expressed in the world frame
        axis - a 3-element numpy array containing the axis of the rotation from
        the current frame to the end effector frame. The magnitude of this vector
        must be sin(angle), where angle is the angle of rotation around this axis
        """

        ## STUDENT CODE STARTS HERE
        # displacement vector
        displacement = target[:3, 3] - current[:3, 3]

        # skew symmetric part of the relative rotation matrix
        R = target[:3, :3] @ current[:3, :3].T

        S = 1 / 2 * (R - R.T)

        # axis vector
        avec = np.array([S[2, 1], S[0, 2], S[1, 0]])
        axis = R @ avec

        ## END STUDENT CODE

        return displacement, axis

    @staticmethod
    def distance_and_angle(G, H):
        """
        Helper function which computes the distance and angle between any two
        transforms.

        This data can be used to decide whether two transforms can be
        considered equal within a certain linear and angular tolerance.

        Be careful! Using the axis output of displacement_and_axis to compute
        the angle will result in incorrect results when |angle| > pi/2

        INPUTS:
        G - a 4x4 numpy array representing some homogenous transformation
        H - a 4x4 numpy array representing some homogenous transformation

        OUTPUTS:
        distance - the distance in meters between the origins of G & H
        angle - the angle in radians between the orientations of G & H


        """

        ## STUDENT CODE STARTS HERE
        # distance
        G_origin = G[:3, 3]
        H_origin = H[:3, 3]
        distance = np.linalg.norm(G_origin - H_origin)

        # angle between orientations
        R = G[:3, :3] @ H[:3, :3].T
        trace = np.trace(R)
        angle = acos(np.clip((trace - 1) / 2, -1, 1))
        ## END STUDENT CODE

        return distance, angle

    def is_valid_solution(self, q, target):
        """
        Given a candidate solution, determine if it achieves the primary task
        and also respects the joint limits.

        INPUTS
        q - the candidate solution, namely the joint angles
        target - 4x4 numpy array representing the desired transformation from
        end effector to world

        OUTPUTS:
        success - a Boolean which is True if and only if the candidate solution
        produces an end effector pose which is within the given linear and
        angular tolerances of the target pose, and also respects the joint
        limits.
        """

        ## STUDENT CODE STARTS HERE
        # check if the joint angles are within the joint limits
        within_limits = [self.lower[i] <= q[i] <= self.upper[i] for i in range(7)]

        if not all(within_limits):
            print("NOT WITHIN JOINT LIMITS: ", within_limits)
            return False

        # end effector pose
        fk = FK()
        _, T_current = fk.forward(q)

        # displacement and axis between the achieved and target end effector poses

        distance, angle = self.distance_and_angle(target, T_current)

        # checking tolerances
        within_linear_tol = distance <= self.linear_tol
        within_angular_tol = abs(angle) <= self.angular_tol
        if not within_angular_tol: print('NOT WITHIN ANGULAR TOL: ', angle)
        if not within_linear_tol: print("NOT WITHIN LINEAR TOL: ", distance)
        success = within_limits and within_linear_tol and within_angular_tol

        ## END STUDENT CODE

        return success

    ####################
    ## Task Functions ##
    ####################

    @staticmethod
    def end_effector_task(q, target):
        """
        Primary task for IK solver. Computes a joint velocity which will reduce
        the error between the target end effector pose and the current end
        effector pose (corresponding to configuration q).

        INPUTS:
        q - the current joint configuration, a "best guess" so far for the final answer
        target - a 4x4 numpy array containing the desired end effector pose

        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """

        ## STUDENT CODE STARTS HERE
        fk = FK()
        _, current = fk.forward(q)
        disp, axis = IK.displacement_and_axis(target, current)
        dq = IK_velocity(q, disp, axis)

        ## END STUDENT CODE

        return dq

    @staticmethod
    def joint_centering_task(q, rate=5e-1):
        """
        Secondary task for IK solver. Computes a joint velocity which will
        reduce the offset between each joint's angle and the center of its range
        of motion. This secondary task acts as a "soft constraint" which
        encourages the solver to choose solutions within the allowed range of
        motion for the joints.

        INPUTS:
        q - the joint angles
        rate - a tunable parameter dictating how quickly to try to center the
        joints. Turning this parameter improves convergence behavior for the
        primary task, but also requires more solver iterations.

        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """

        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # normalize the offsets of all joints to range from -1 to 1 within the allowed range
        offset = 2 * (q - IK.center) / (IK.upper - IK.lower)
        dq = rate * -offset  # proportional term (implied quadratic cost)

        return dq

    ###############################
    ## Inverse Kinematics Solver ##
    ###############################

    def inverse(self, target, seed):
        """
        Uses gradient descent to solve the full inverse kinematics of the Panda robot.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        seed - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], which
        is the "initial guess" from which to proceed with optimization

        OUTPUTS:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], giving the
        solution if success is True or the closest guess if success is False.
        success - True if the IK algorithm successfully found a configuration
        which achieves the target within the given tolerance. Otherwise, False
        rollout - a list containing the guess for q at each iteration of the algorithm
        """

        q = seed
        rollout = []

        while True:

            rollout.append(q)

            # Primary Task - Achieve End Effector Pose
            dq_ik = self.end_effector_task(q, target)

            # Secondary Task - Center Joints
            dq_center = self.joint_centering_task(q)

            ## STUDENT CODE STARTS HERE
            J = calcJacobian(q)
            Jns = null_space(J)

            # Task Prioritization

            # primary task
            dq = dq_ik

            # projection/secondary task
            dq += np.reshape(np.matmul(dq_center, Jns) / linalg.norm(Jns)**2 * Jns, (7,))

            # Termination Conditions
            if (len(rollout) >= self.max_steps) or (np.linalg.norm(dq) <= self.min_step_size):
                break


            ## END STUDENT CODE

            q = q + dq

        success = self.is_valid_solution(q, target)
        return q, success, rollout


################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True, precision=5)

    ik = IK()
    fk = FK()

    # matches figure in the handout
    seed = np.array([0, 0, 0, -pi / 2, 0, pi / 2, pi / 4])

    q_in = np.array([-1, .2, 0, -pi / 2, 0, 2, pi / 4])
    
    target = np.array([
        [0, -1, 0, 0.3],
        [-1, 0, 0, 0],
        [0, 0, -1, .8],
        [0, 0, 0, 1],
        ])

    q, success, rollout = ik.inverse(target, seed)


    _, T0e = fk.forward(q)
    for i, q in enumerate(rollout):
        joints, pose = ik.fk.forward(q)
        d, ang = IK.distance_and_angle(target, pose)
        print('iteration:', i, ' q =', q, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d, ang=ang))

    print("Success: ", success)
    print("Solution: ", q)
    print("Iterations:", len(rollout))
    print("Expected:")
    print(target)
    print("Actual:")
    print(T0e)
