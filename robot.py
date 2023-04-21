from core.interfaces import ArmController
from core.interfaces import ObjectDetector
from vision import *
import numpy as np
from math import sin, pi, cos
from calculateFK import FK
from solveIK import IK


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
    
    def move_above_block(self, H_w_block):
        """
        :param H_w_block: 4x4 transformation of block in world frame
        """
        seed = self.best_seed(H_w_block[:3, -1])
        target = H_w_block

        # TODO: align axes properly and set the goal height above the block
        goal_orientation = self.fk.forward(self.get_positions())[1][:3, :3]
        target[:3, :3] = goal_orientation

        # going above target block by some above_height
        target[2, 3] = target[2, 3] + self.above_height
        

        # TODO: plan a path from q_curr to q_out and execute that path

        q_out, success, _ = self.ik.inverse(target, seed)

        if (success):
            self.safe_move_to_position(q_out)
            print("Moving to above block")
        else: print("Solution to above block not found")

    def get_static_block(self):

        # setting new position
        q_curr = self.get_positions()
        _, T0e = self.fk.forward(q_curr)
        target = T0e
        target[2, 3] -= self.above_height
        seed = self.best_seed(target[:3, -1])

        # opening gripper then move down
        self.open_gripper()
        q_down, success, _ = self.ik.inverse(target, seed)
        if success:
            self.safe_move_to_position(q_down)
            print("Grabbing block...")
        else: print("Solution to grab block not found...")

        # closing the gripper then moving up
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
        z_stack = .225 + .05 * self.num_blocks_stacked
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
            self.safe_move_to_position(q_stack)
            print("Moving to stack...")
        else: print('Cannot find path to stack...')

        # dropping the block and moving out of the way
        success = self.open_gripper()
        if success: print("Dropping block...")
        self.num_blocks_stacked += 1
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
        gripper_status = self.exec_gripper_cmd(.01, 50)
        if gripper_status: print("Closing gripper...")




if __name__ == "__main__":
    robot = Robot('red')
    a = robot.best_seed(np.array([0, 0, 1]))
    print(a)