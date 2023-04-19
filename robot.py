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

    # CURRENT states
    q_curr = 0
    grip_state = 0
    
    def __init__(self):
        super().__init__()
        self.fk = FK()
        self.ik = IK()
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
        above_height = .1

        # TODO: align axes properly and set the goal height above the block
        goal_orientation = self.fk.forward(self.get_positions())[1][:3, :3]
        target[:3, :3] = goal_orientation
        target[2, 3] = target[2, 3] + above_height
        

        # TODO: plan a path from q_curr to q_out and execute that path

        q_out, success, _ = self.ik.inverse(target, seed)

        if (success):
            self.safe_move_to_position(q_out)
            print("Moving to above block")
        else: print("Solution to above block not found")

    def get_static_block(self):
        # assume we are oriented above a block and in the right direction

        # TODO: move down by some amount

        # TODO: close gripper with some force

        return

if __name__ == "__main__":
    robot = Robot()
    a = robot.best_seed(np.array([0, 0, 1]))
    print(a)