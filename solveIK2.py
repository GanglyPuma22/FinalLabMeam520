import numpy as np
from math import pi, acos
from scipy.linalg import null_space

from lib.calcJacobian import calcJacobian
from lib.calculateFK import FK
from lib.IK_velocity import IK_velocity

class IK:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK()

    def __init__(self,linear_tol=1e-4, angular_tol=1e-3, max_steps=500, min_step_size=1e-5):
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


        curr_pos = current[:3, 3]
        target_pos = target[:3, 3]

        curr_pos = np.reshape(curr_pos, (3,1))
        target_pos = np.reshape(target_pos, (3,1))


        displacement = target_pos - curr_pos

        # print(displacement)

        curr_rot = current[:3,:3]
        target_rot = target[:3,:3]
        R = np.transpose(curr_rot) @ target_rot

        S = 0.5*(R - np.transpose(R))
        a = np.array((S[2,1],S[0,2],S[1,0]))
        # print(a)
        # print(target_rot)

        axis = curr_rot @ a
        # print(axis)
        #
        # disp_unit = displacement / np.linalg.norm(displacement)
        # a2 = np.cross(a, disp_unit)
        # a2 = a2 / np.linalg.norm(a2)
        #
        # # sin(theta) is equal to the magnitude of a, which is what we want to be the magnitude of a2 as well
        # sin_theta = np.linalg.norm(a)
        #
        # axis = a2 * sin_theta

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

        # ik = IK()
        #
        # d, a = ik.distance_and_angle(G, H)
        G_pos = G[:3,3]
        H_pos = H[:3,3]

        G_pos = np.reshape(G_pos, (3,1))
        H_pos = np.reshape(H_pos, (3,1))

        disp = G_pos - H_pos
        distance = np.linalg.norm(disp)

        G_R = G[:3, :3]
        H_R = H[:3, :3]

        R = np.transpose(G_R) @ H_R
        trace = np.trace(R)
        input = np.clip((trace-1)*0.5, -1, 1)
        angle = acos(input)

        ## END STUDENT CODE

        return distance, angle

    def is_valid_solution(self,q,target):
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

        success = True
        ## STUDENT CODE STARTS HERE
        # if (q[0] < -2.8973 or q[0] > 2.8973 or q[1] < -1.7628 or q[1] > -1.7628 or q[2] < -2.8973 or q[2] > 2.8973
        # or q[3] > -0.0698 or q[3] < -3.0718 or q[4] < -2.8973 or q[4] > 2.8973 or q[5] < -0.0175 or q[5] > 3.7525 or
        # q[6] < -2.8973 or q[6] > 2.8973):
        #     return False

        for i in range(len(q)):
            if q[i] > self.upper[i] or q[i] < self.lower[i]:
                print("bruh moment")
                return False

        # ik = IK()
        _, T0e = self.fk.forward(q)
        d, a = self.distance_and_angle(target, T0e)

        lt = self.linear_tol
        at = self.angular_tol

        if (d > lt):
            print("L eric")
            return False

        if (a > at):
            print("uh oh")
            return False

        ## END STUDENT CODE

        return success

    ####################
    ## Task Functions ##
    ####################

    @staticmethod
    def end_effector_task(q,target):
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

        ik = IK()
        _, curr = ik.fk.forward(q)
        d, a = ik.displacement_and_axis(target, curr)
        d = np.reshape(d, (3,))
        a = np.reshape(a, (3,))
        # q = np.reshape(q, (1,7))
        print(d.shape)
        print(a.shape)
        print(q.shape)
        dq = IK_velocity(q, d, a)


        ## END STUDENT CODE

        return dq

    @staticmethod
    def joint_centering_task(q,rate=5e-1):
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
        dq = rate * -offset # proportional term (implied quadratic cost)

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
        which achieves the target within the given tolerance. Otherwise False
        rollout - a list containing the guess for q at each iteration of the algorithm
        """

        q = seed
        rollout = []


        steps = 0


        while True:

            rollout.append(q)

            # Primary Task - Achieve End Effector Pose
            dq_ik = self.end_effector_task(q,target)
            print(dq_ik)

            # Secondary Task - Center Joints
            dq_center = self.joint_centering_task(q)

            ## STUDENT CODE STARTS HERE

            # Task Prioritization
                # v = np.linalg.solve(J, 0)
            # print(J)
            # print(np.transpose(J))

            J = calcJacobian(q)
            ns = null_space(J)
            # proj = (np.dot(dq_center, ns) / ((np.linalg.norm(ns))**2)) * ns
            proj = np.dot(dq_center, ns) * ns

            proj= np.reshape(proj, (1, 7))

            dq_ik = dq_ik.reshape((1, 7))
            dq = dq_ik + proj
            # dq = dq.flatten()
            # print(dq)



            # Termination Conditions
            if (len(rollout) > self.max_steps or np.linalg.norm(dq) < self.min_step_size): # TODO: check termination conditions
                break # exit the while loop if conditions are met!

            ## END STUDENT CODE

            q = q + dq
            empty_q = np.zeros(7)
            empty_q[:] = q[0, :]
            q = empty_q.copy()
            steps += 1
            # import ipdb; ipdb.set_trace()

        success = self.is_valid_solution(q,target)
        return q, success, rollout

################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    ik = IK()

    # matches figure in the handout
    seed = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    target = np.array([
        [0,-1,0,0.3],
        [-1,0,0,0],
        [0,0,-1,.5],
        [0,0,0, 1],
    ])

    q, success, rollout = ik.inverse(target, seed)

    for i, q in enumerate(rollout):
        joints, pose = ik.fk.forward(q)
        d, ang = IK.distance_and_angle(target,pose)
        print('iteration:',i,' q =',q, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d,ang=ang))

    print("Success: ",success)
    print("Solution: ",q)
    print("Iterations:", len(rollout))
