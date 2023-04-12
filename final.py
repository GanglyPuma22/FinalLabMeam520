import sys
import numpy as np
from copy import deepcopy
from math import pi
from solveIK import IK 
from calculateFK import FK
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

    #Predifined position where we can see the blocks 
    view_block_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353 + pi/5, 0.75344866])
    #Move from start to see the blocks
    arm.safe_move_to_position(view_block_position)

    # get the transform from camera to panda_end_effector
    H_ee_camera = detector.get_H_ee_camera()
    print(H_ee_camera)
    print(detector.get_detections())

    # Detect some blocks...
    for (name, pose) in detector.get_detections():
         print(name,'\n',pose)
    
    #Get transform from camera to worldframe and offset it to end effector
    fk = FK()
    ik = IK()

    jointPos, T0e = fk.forward(start_position)

    #Camera frame expressed in world coordinates
    print(T0e)
    T0c = T0e @ H_ee_camera #This seems right compared to end effector pose
    print(T0c)

    if len(detector.get_detections()) > 0:
        #Rotation matrix of block in camera frame
        TcB = detector.get_detections()[0][1]
        print(TcB)
        #Turn it into world frame coordinates
        T0B = T0c @ TcB
        print(T0B)
        #Add 10 cm to the blocks z pose to define target pose for end effector
        T0B[2,3] = T0B[2,3] + 0.1

        #Use IK to get joint angles we need to move the robot above the block
        blockHoverConfig = ik.inverse(T0B, view_block_position)
        print(blockHoverConfig)
        arm.safe_move_to_position(blockHoverConfig)




    

    #We now have the position and orientation of the target blocks center.
    #Use IK to get a goal position that is above the block by 10cm 

    #Path planning to goal
    #Iterate over path and safe move to each config
    
    #Orient the end effector to be parallele with block

    #Go down 10 cm in a straight line -> Look at line task from previous lab

    #close gripper 
    
    # go up to safe position

    # END STUDENT CODE