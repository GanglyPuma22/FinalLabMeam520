import sys
import numpy as np
from copy import deepcopy
from math import pi
from solveIK import IK 
from calculateFK import FK
from calcJacobian import calcJacobian
from robot import Robot
from vision import *
import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds


if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    # STUDENT CODE HERE

    robot = Robot(team)
    fk = FK()
    ik = IK()

    # predifined position where we can see the blocks 
    view_block_position = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    robot.safe_move_to_position(view_block_position)

    # locating blocks in world frame
    _, T0e = fk.forward(robot.get_positions())
    H_w_cam = H_w_cam(T0e, detector.get_H_ee_camera())
    poses = []
    for (block_name, pose) in detector.get_detections():
        poses.append(pose)
    poses = objects_in_world(poses, H_w_cam)
    
    # stacking each static block
    for H in poses:
        print('Block: ', H)
        robot.move_above_block(H)
        robot.get_static_block()
        robot.stack_static_block()

    if team == 'red':
        robot.get_dynamic_block()
        robot.get_dynamic_block()
        robot.get_dynamic_block()
        robot.get_dynamic_block()
        robot.get_dynamic_block()
        robot.get_dynamic_block()
    # END STUDENT CODE