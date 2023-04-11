import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
from lib.calcJacobian import calcJacobian
from lib.calculateFK import FK
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap


class PotentialFieldPlanner:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK()

    def __init__(self, tol=4e-3, max_steps=2500, min_step_size=0.2):
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint sets
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # YOU MAY NEED TO CHANGE THESE PARAMETERS

        # solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation 

    @staticmethod
    def attractive_force(target, current):
        """
        Helper function for computing the attactive force between the current position and
        the target position for one joint. Computes the attractive force vector between the 
        target joint position and the current joint position 

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint 
        from the current position to the target position 
        """

        ## STUDENT CODE STARTS HERE


        att_f = np.zeros((3,))
        
        #Push joints towards target pose
        att_f = -1*(current-target)

        ## END STUDENT CODE
        #print(np.shape(att_f))
        #print(scallableF)
        return att_f

    @staticmethod
    def repulsive_force(obstacle, current, unitvec=np.zeros((3,1))):
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the 
        obstacle and the current joint position 

        INPUTS:
        obstacle - 1x6 numpy array representing the an obstacle box in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position 
        to the closest point on the obstacle box 

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint 
        from the obstacle
        """

        ## STUDENT CODE STARTS HERE
        dist, unit = PotentialFieldPlanner.dist_point2box([current.T], obstacle)
        dist = dist.flatten()
        distanceTol = 0.2
        rep_f = np.zeros((3, )) 
        b = -dist*unit[0].T
        #n=0.005 #not sure what this param does
        n= 0.01

        #Check to make sur we are at a safe distance
        if dist > 0:
            if dist - distanceTol < 0: #We are not at a safe distance so repulsive force not 0
                rep_f = n*((1/dist)-(1/distanceTol)) * (1/(dist*dist)) * ((b)/np.linalg.norm(b))

        ## END STUDENT CODE
        return rep_f

    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces. Computes the closest point
        on the box to a given point 
    
        INPUTS:
        p - nx3 numpy array of points [x,y,z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
                dist > 0 point outside
                dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector 
        from the point to the closest spot on the box
            norm(unit) = 1 point is outside the box
            norm(unit)= 0 point is on/inside the box

         Method from MultiRRomero
         @ https://stackoverflow.com/questions/5254838/
         calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # Get box info
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin*0.5 + boxMax*0.5
        p = np.array(p)

        # Get distance info from point to box boundary
        dx = np.amax(np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T, 1)
        dy = np.amax(np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T, 1)
        dz = np.amax(np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T, 1)

        # convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        # Figure out the signs
        signs = np.sign(boxCenter-p)

        # Calculate unit vector and replace with
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    @staticmethod
    def compute_forces(target, obstacle, current):
        """
        Helper function for the computation of forces on every joints. Computes the sum 
        of forces (attactive, repulsive) on each joint. 

        INPUTS:
        target - 3x7 numpy array representing the desired joint/end effector positions 
        in the world frame
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x7 numpy array representing the current joint/end effector positions 
        in the world frame

        OUTPUTS:
        joint_forces - 3x7 numpy array representing the force vectors on each 
        joint/end effector
        """

        ## STUDENT CODE STARTS HERE

        joint_forces = np.zeros((3, 7)) 
        
        for i in range(0,7):
            att = PotentialFieldPlanner.attractive_force(target[:,i],current[:,i])
            rep = np.zeros((3,))
            for j in range(np.shape(obstacle)[0]):
                rep = rep + PotentialFieldPlanner.repulsive_force(obstacle[j,:], current[:,i])
            
            #print(att)
            #print(rep)
            # print(att+rep)
            #print("Att force: " + str(np.linalg.norm(att)))
            #print(np.linalg.norm(att))
            #print("Rep force: " + str(np.linalg.norm(rep)))
            joint_forces[:,i] = att + rep
            #joint_forces[:,i] = PotentialFieldPlanner.attractive_force(target[i,:],current[i,:]) + PotentialFieldPlanner.repulsive_force(obstacle, current[i,:])
        ## END STUDENT CODE
        return joint_forces
    
    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques. Computes the sum 
        of torques on each joint.

        INPUTS:
        joint_forces - 3x7 numpy array representing the force vectors on each 
        joint/end effector
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x7 numpy array representing the torques on each joint 
        """

        ## STUDENT CODE STARTS HERE

        joint_torques = np.zeros((1, 7))
        #J = calcJacobian(q)
        #print(J)
        #Jv = J[:-3,:] #Keep only first three rows of Jacobian
        #print(Jv)
        currentJ = np.zeros((3,7))

        # Iteratively construct the right Jacobian matrix columns to multiply with the right jointPos force
        for i in range(0, 7):
            #currentJ[:,i] = Jv[:,i]
            Jv = calcJacobian(q, i+1)[:-3,:]#Keep only first three rows of Jacobian
            currentJ = np.zeros((3,7))
            #joint_torques = np.zeros((1, 7))
            for j in range(0, i):
                currentJ[:,j] = Jv[:,j]
            #print(Jv)
            
            # print(currentJ)
            # print(joint_forces[:,i])
            # print(currentJ.T @ joint_forces[:,i])
            joint_torques = joint_torques + currentJ.T @ joint_forces[:,i]
            # print(currentJ)
            # print(joint_forces[:,i])
            #joint_torques = joint_torques + currentJ.T @ joint_forces[:,i]
        ## END STUDENT CODE

        return joint_torques.flatten()

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two
        vectors.

        This data can be used to decide whether two joint sets can be
        considered equal within a certain tolerance.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets 

        """

        ## STUDENT CODE STARTS HERE

        distance = np.linalg.norm(target[:-1] - current[:-1])

        ## END STUDENT CODE

        return distance
    
    @staticmethod
    def compute_gradient(q, target, map_struct, alpha):
        """
        Computes the joint gradient step to move the current joint positions to the
        next set of joint positions which leads to a closer configuration to the goal 
        configuration 

        INPUTS:
        q - 1x7 numpy array. the current joint configuration, a "best guess" so far for the final answer
        target - 1x7 numpy array containing the desired joint angles
        map_struct - a map struct containing the obstacle box min and max positions

        OUTPUTS:
        dq - 1x7 numpy array. a desired joint velocity to perform this task
        """

        ## STUDENT CODE STARTS HERE
        #Get current and target joint Pose arrays
        current, _ = PotentialFieldPlanner.fk.forward(q)
        target, _ = PotentialFieldPlanner.fk.forward(target)

        # #Remove last row corresponding to end effector pose
        # current = current[:-1,:].T
        # target = target[:-1,:].T

        #Remove first row so that joint 1 doesnt impact forces
        current = np.delete(current, (0), axis=0).T
        target = np.delete(target, (0), axis=0).T

        forces = PotentialFieldPlanner.compute_forces(target, map_struct.obstacles, current)
        #print(forces)
        torques = PotentialFieldPlanner.compute_torques(forces, q)
        #print(torques)
        #if np.linalg.norm(torques) == 0:
            #print("NORM OF TORQUES IS ZERO: ")
            #print(forces)
            #print(torques)

        #Compute update for gradient descent
        dq = (alpha/np.linalg.norm(torques)) * torques
        # print(q)
        # print(dq)
        # print(q + dq)
        ## END STUDENT CODE
        return dq

    ###############################
    ### Potential Feild Solver  ###
    ###############################

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from the startng configuration to
        the goal configuration.

        INPUTS:
        map_struct - a map struct containing min and max positions of obstacle boxes 
        start - 1x7 numpy array representing the starting joint angles for a configuration 
        goal - 1x7 numpy array representing the desired joint angles for a configuration

        OUTPUTS:
        q - nx7 numpy array of joint angles [q0, q1, q2, q3, q4, q5, q6]. This should contain
        all the joint angles throughout the path of the planner. The first row of q should be
        the starting joint angles and the last row of q should be the goal joint angles. 
        """

        #q_path = np.array([]).reshape(0,7)
        q_path = start
        q = start
        iterNum = 1
        lastDq = PotentialFieldPlanner.compute_gradient(q, goal, map_struct, 0.09)

        while True:

            ## STUDENT CODE STARTS HERE
            
            # The following comments are hints to help you to implement the planner
            # You don't necessarily have to follow these steps to complete your code 
            
            # Compute gradient 
            dq = PotentialFieldPlanner.compute_gradient(q, goal, map_struct, 0.09)
            q = q + dq
            
            # YOU MAY NEED TO DEAL WITH LOCAL MINIMA HERE
            # TODO: when detect a local minima, implement a random walk
            if iterNum > 10:
                # print(q)
                # print(q_path[iterNum - 2])
                #print(self.q_distance(q, q_path[iterNum - 2]))
                localDist = 0
                distNum = 10
                #Compute average distance between last distNum configurations
                for i in range(1, distNum + 1):
                    localDist += self.q_distance(q, q_path[iterNum-i])
                
                
                #print(localDist/distNum)
                if localDist/distNum < 0.07: #minimum average distance from last 10 configs before we start random walk
                    walkStartq = q
                    #print("RANDOM WALK NEEDED FOR ITER" + str(iterNum))
                    walkSteps = 0
                    #Continue walking while new q isnt far enough or there is collision
                    while (self.q_distance(walkStartq, q) < 0.45 or isCollision(q, map_struct)) and walkSteps < 50:
                        #print(q)
                        #print(self.q_distance(walkStartq, q))
                        q = q + np.random.uniform(low=-0.05, high=0.05, size=7)
                        walkSteps += 1

                    #print("Final q: ")
                    #print(q)    
                    #print("Final distance: " + str(self.q_distance(walkStartq, q)))                

                # if self.q_distance(q, q_path[iterNum - 2]) < self.tol:
                #     #q = q + np.array
                #     print("RANDOM WALK NEEDED FOR ITER" + str(iterNum))
                #     q = q + np.random.uniform(low=-0.08, high=0.08, size=7)
                #     print(q)
            # YOU NEED TO CHECK FOR COLLISIONS WITH OBSTACLES

            #Check joint limits
            for i in range(0, len(q)):
                    q[i] = np.clip(q[i], PotentialFieldPlanner.lower[i], PotentialFieldPlanner.upper[i])

            if isCollision(q, map_struct):
                print("OBSTACLE COLLISION DETECTED")
                return q_path
            
            #No collision
            #Add new joint angles to path
            q_path = np.vstack([q_path, q])

            # Termination Conditions, distance between current joint configs and goal is below tolerance
            if PotentialFieldPlanner.q_distance(goal, q) < self.min_step_size or iterNum > self.max_steps: 
                break # exit the while loop if conditions are met!

            iterNum += 1
            #print(np.linalg.norm(lastDq - dq))
            #lastDq = dq
            ## END STUDENT CODE
        q_path = np.vstack([q_path, goal])

        
        return q_path

################################
## Simple Testing Environment ##
################################
def isCollision(q, map):
    # #Check jointPoses for collisions
    jointPos, _ = PotentialFieldPlanner.fk.forward(q)
    # Create linePt arrays corresponding to robot links
    p1 = jointPos[0]
    p2 = jointPos[1]
    for i in range(1,7):
        newP1 = jointPos[i]
        newP2 = jointPos[i+1]
        p1 = np.vstack([p1, newP1])
        p2 = np.vstack([p2, newP2])

    #Check if any link intersects with any obstacle. Links treated as lines.
    for i in range(len(map.obstacles)):
        collisions = detectCollision(p1, p2, map.obstacles[i])
        if np.sum(np.array(collisions).T.astype(int)) > 0:
            return True
    
    return False
if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    planner = PotentialFieldPlanner()
    
    #planner.attractive_force(np.array([6, 4, 2]).T, np.array([2, 2, 0]).T)
    
    # target = np.array([[6, 4, 2],[6, 4, 2],[6, 4, 2],[6, 4, 2],[6, 4, 2],[6, 4, 2],[6, 4, 2]])
    # current = np.array([[2, 2, 0],[2, 2, 0],[2, 2, 0],[2, 2, 0],[2, 2, 0],[2, 2, 0],[2, 2, 0]])
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])

    # inputs 
    map_struct = loadmap("../maps/map1.txt")
    #print(map_struct.obstacles[0])
    #planner.repulsive_force(map_struct.obstacles[0], np.array([2, 2, 0]).T)
    # target, _ = PotentialFieldPlanner.fk.forward(goal)
    # current, _ = PotentialFieldPlanner.fk.forward(start)
    # current = current[:-1,:].T
    # target = target[:-1,:].T

    # forces = planner.compute_forces(target, map_struct.obstacles, current)
    
    # torques = planner.compute_torques(forces, start)
    # grad = planner.compute_gradient(start, goal, map_struct, 0.1)

    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    minError = 1000
    # show results
    for i in range(q_path.shape[0] - 1):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))
        if error < minError:
            minError = error

    print("q path: ", q_path)
    print("Min error: " + str(error))
    #print("Goal:")
    #print(goal)