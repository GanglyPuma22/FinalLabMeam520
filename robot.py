from core.interfaces import ArmController
from core.interfaces import ObjectDetector
from vision import *
import numpy as np
from math import sin, pi, cos
from calculateFK import FK
from solveIK import IK
from follow import JacobianDemo


class Robot(ArmController):

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])
    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint

    # properties
    above_height = .1
    num_blocks_stacked = 0
    
    def __init__(self, team):
        super().__init__()
        self.fk = FK()
        self.ik = IK()
        self.team = team
        self.detector = ObjectDetector()
        self.JD = JacobianDemo()
        pass

    @staticmethod
    def best_seed(T):
        """
        :param T: [3,] location you want to go to
        :returns the best seed to use to get to this position
        """
        # TODO: get a more expansive list of various seeds
        seeds = [np.array([0,0,0,-pi/2,0,pi/2,pi/4])]

        # TODO: choose the seed that has the end effector at the closes position to T

        return seeds[0]
    
    @staticmethod
    def orient_to_static2(H_w_block):
        # TODO: address edge cases of being outside of joint limits
        R_w_ee = np.array([[1, 0, 0],
                           [0, -1, 0],
                           [0, 0, -1]])
        R_w_block = H_w_block[:3, :3]
        R_ee_block = R_w_block @ R_w_ee.T

        col_mask = np.abs(np.round(R_ee_block[-1], 4)) != 1
        H = (R_ee_block.T[col_mask]).T
        ee_x = np.array([0, 1, 0])
        dots = H.T @ ee_x
        max_dot = np.argmax(dots)
        tx = H[:, max_dot]
        tz = np.array([0, 0, -1])
        ty = np.cross(tz, tx)
        target = np.vstack([tx, ty, tz]).T
        theta = np.arccos(target[0][0])
        print('Angle: ', theta)
        print("target: ,", target)
        return target


    
    @staticmethod
    def orient_to_static(H_w_block):
        """
        :param H_w_block: [4x4] location transformation of static block in world frame
        :returns the necessary end effector orientation to grab the block
        """
        ee_x = np.array([1, 0, 0])
        
        # removing axis that points up
        H = np.copy(H_w_block[:3, :3])
        col_mask = np.abs(np.round(H[-1], 4)) != 1
        H = (H.T[col_mask]).T
        

        try1 = np.cross(H[:, 0], H[:, 1])
        tz = np.zeros(3)
        tx = np.zeros(3)
        ty = np.zeros(3)
        
        if (try1[-1] == -1):
            tx = H[:, 0]
            ty = H[:, 1]
            tz = np.array([0, 0, -1])
        else:
            ty = H[:, 0]
            tx = H[:, 1]
            tz = np.array([0, 0, -1])

        theta = np.arccos(tx[0])
        print('COS', np.cos(theta))
        print(theta)
 
        # getting the columns of the target
        target = np.vstack([tx, ty, tz]).T

        print('block: ', H)
        print('target: ', target)
        return target
    
    def move_above_block(self, H_w_block):
        """
        :param H_w_block: 4x4 transformation of block in world frame
        """
        seed = self.best_seed(H_w_block[:3, -1])
        target = H_w_block
        target[:3, :3] = self.orient_to_static2(H_w_block)

        # going above target block by some above_height
        target[2, 3] += self.above_height

        q_out, success, _ = self.ik.inverse(target, seed)

        if (success):
            self.safe_move_to_position(q_out)
            print("Moving to above block")
        else: print("Solution to above block not found")

    
    def control_joint_vel(self):
        """
        Calls jacobian demo class on arm controller super class to use arm.safe_set_joint_positions_velocities
        """
        #TODO: Fix line trajectory function in follow.py
        state = self.get_state()
        self.JD.activate()
        print("STATE IS:")
        print(state)
        self.JD.line_trajectory(state, super())


    def get_angle(self, goal, block):
         
        """
        Function tries two ways of computign angle in xy plane between end effector and block
        :param goal: 4x4 transformation of end effector goal in world frame
        :param block: 4x4 transfromation of block in world frame
        """
        endEffectorX = np.transpose(goal[:3,0])
        blockX = np.transpose(block[:3, 0])

        #Dot product formula for angle
        angle = np.arccos(np.dot(endEffectorX, blockX)/ 
                          (np.linalg.norm(endEffectorX) * np.linalg.norm(blockX)))
        
        distance, angle = self.ik.distance_and_angle(goal, block)

        #Change target so x and y axes between end effector and block are aligned
        goal[:3, 0] = np.transpose(np.array([np.cos(angle), np.sin(angle), 0]))
        goal[:3, 1] = np.transpose(np.array([-np.sin(angle), np.cos(angle), 0]))

        return angle, goal

    def get_static_block(self):

        # setting new position
        q_curr = self.get_positions()
        _, T0e = self.fk.forward(q_curr)
        target = T0e
        target[2, 3] -= self.above_height
        seed = self.best_seed(target[:3, -1])

        # opening gripper then move down
        #TODO: Change moving down to line trajectory
        self.open_gripper()
        q_down, success, _ = self.ik.inverse(target, seed)
        if success:
            self.safe_move_to_position(q_down)
            print("Grabbing block...")
        else: print("Solution to grab block not found...")

        # closing the gripper then moving up
        #TODO: Change moving up to line trajectory
        gripper_status = self.exec_gripper_cmd(.05, 50)
        if gripper_status: print("Closing gripper...")
        if success:
            self.safe_move_to_position(q_curr)
            print("Lifting block...")
        else: print("Solution to lift block not found...")

    def stack_static_block(self):
        # setting the desired location to stack the next block
        x_stack = .59 
        y_stack = .23 if (self.team=='red') else -.23
        z_stack = .23 + .05 * self.num_blocks_stacked
        target = np.array([[1, 0, 0, x_stack],
                            [0, -1, 0, y_stack],
                            [0, 0, -1, z_stack],
                            [0, 0, 0, 1]])
        seed = self.best_seed(np.array(target[:3, -1]))
        
        # TODO: make path to stack
        # moving to just above the desired location
        above_target = np.copy(target)
        above_target[2, -1] += self.above_height
        q_above, success, _ = self.ik.inverse(above_target, seed)
        if success:
            self.safe_move_to_position(q_above)
            print('Going above block stack...')
        else: print('Cannot find path above the stack...')

        # moving down to stack
        q_stack, success, _ = self.ik.inverse(target, self.get_positions())
        if success:
            #TODO: Replace move down with line trajectory
            self.safe_move_to_position(q_stack)
            print("Moving to stack...")
        else: print('Cannot find path to stack...')

        # dropping the block and moving out of the way
        success = self.open_gripper()
        if success: print("Dropping block...")
        self.num_blocks_stacked += 1
        #TODO: Replace move up with line trajectory
        self.safe_move_to_position(q_above)

    def get_dynamic_block(self):
        #Seed where camera is pointing down y axis of world and end effector is just above and at the edge of dynamic table
        dynamic_seed = [ 0.75395863,  0.65412359,  0.69980793, -1.9420197,   1.78589341,  2.27965587, 1.19368807]
        #dynamic_seed_side= np.array([ 0.86509697, 0.9800777, 0.57962444, -1.3424143, 0.98768188, 1.5608464, 1.77350871])

        ee_Z = 0.22
        ee_X = 0
        ee_Y = 0.748
        sideTarget = np.array([[0, 0, -1, ee_X], [0, -1, 0, ee_Y], [-1, 0, 0, ee_Z], [0, 0, 0, 1]])
        targetCameraFront = np.array([[0, 1, 0, ee_X], [0, 0, 1, ee_Y - 0.05], [1, 0, 0, ee_Z], [0, 0, 0, 1]])
        print(targetCameraFront)

        #Solve for path that gets us to camera front position
        q_out, success, _ = self.ik.inverse(targetCameraFront, dynamic_seed)
        print(q_out)

        #if (success):
        print("Moving to start dynamic block task position")
        self.safe_move_to_position(q_out)
        gripper_status = self.exec_gripper_cmd(.09, 50)
        if gripper_status: print("Opening gripper...")
        blockTarget = np.array([[0, 1, 0, ee_X], [0, 0, 1, 0.75], [1, 0, 0, ee_Z], [0, 0, 0, 1]])
        q_out, success, _ = self.ik.inverse(blockTarget, q_out)
        self.safe_move_to_position(q_out)
        print("Moving to intercept block")
        gripper_status = self.exec_gripper_cmd(.01, 55)
        if gripper_status: print("Closing gripper...")




if __name__ == "__main__":
    robot = Robot('red')
    a = robot.best_seed(np.array([0, 0, 1]))
    print(a)
