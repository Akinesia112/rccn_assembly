import pybullet as p
import time
import math
from datetime import datetime
import pybullet_data

clid = p.connect(p.SHARED_MEMORY)
#clid = p.connect(p.GUI)

if (clid < 0):
    p.connect(p.GUI)

p.setPhysicsEngineParameter(enableConeFriction=0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

planeId = p.loadURDF("plane.urdf", [0, 0, -0.3])

husky = p.loadURDF("husky/husky.urdf", [0.290388, 0.329902, -0.310270],
                   [0.002328, -0.000984, 0.996491, 0.083659], globalScaling = 1.22)

for i in range(p.getNumJoints(husky)):
    print(p.getJointInfo(husky, i))

# Load the new URDF file
    
ra620_1621_path = r"D:\RCCN_Assembly\rccn_assembly\scripts\Data\hiwin_ra610_1476_support\urdf\ra610_1476.urdf"
hiwinId = p.loadURDF(ra620_1621_path, [0, 0, .3], useFixedBase=1)

jointPositions = [3.559609, 0.411182, 0.862129, 1.744441, 0.077299, -1.129685, 0.0, 0.0, 0.0]
for jointIndex in range(p.getNumJoints(hiwinId)):
    p.resetJointState(hiwinId, jointIndex, jointPositions[jointIndex])


#put hiwin on top of husky

cid = p.createConstraint(husky, -1, hiwinId, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0., 0., -.5],
                         [0, 0, 0, 1])

baseorn = p.getQuaternionFromEuler([3.1415, 0, 0.3])
baseorn = [0, 0, 0, 1]
#[0, 0, 0.707, 0.707]

#p.resetBasePositionAndOrientation(hiwinId,[0,0,0],baseorn)#[0,0,0,1])
hiwinEndEffectorIndex = 8
numJoints = p.getNumJoints(hiwinId)

if (numJoints != 9):
  exit()

ll = [5.8, 4, 5.8, 4, 5.8, 4, 4, 4, 4]  # Lower limits
ul = [5.8, 4, 5.8, 4, 5.8, 4, 4, 4, 4]          # Upper limits
jr = [5.8, 4, 5.8, 4, 5.8, 4, 4, 4, 4]                   # Joint ranges
rp = [0, 0, 0, 0.5 , 0, 0.66, 0, 0, 0]  # Rest poses
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]      # Joint damping coefficients


for i in range(numJoints):
  p.resetJointState(hiwinId, i, rp[i])

p.setGravity(0, 0, -10)
t = 0.
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
useNullSpace = 0

useOrientation = 0
#If we set useSimulation=0, it sets the arm pose to be the IK result directly without using dynamic control.
#This can be used to test the IK result accuracy.
useSimulation = 1
useRealTimeSimulation = 0
p.setRealTimeSimulation(useRealTimeSimulation)

target_positions = [[0.8, 0, 0.6], [0.8, 0.4, 0.6], [0.8, 1.6, 1.0], [1.6, 1.6, 1.0], [1.6, 0.8, 1.0], 
                    [1.6, 0.8, 1.2], [1.6, 0, 1.2],[0.8, 0, 1.2], [0.8, 0, 0.6],[1.6, 1.6, 1.6], [1.6, 0.8, 1.6],
                    [1.6, 0.8, 1.8], [1.6, 0, 1.8], [0.8, 0, 1.8], [0.8, 0, 1.6], [0.8, 0.4, 1.6], [0.8, 0.4, 1.8],
                    [-0.8, -1.6, 1.8], [-1.6, -1.6, 1.8], [-1.6, -0.8, 1.8], [-1.6, -0.8, 1.6], [-1.6, 0, 1.6], [-0.8, 0, 1.6],
                    [-0.8, 0, 1.8], [-0.8, 0.4, 1.8], [-0.8, 0.4, 1.6], [-0.8, 0.4, 1.6], [-0.8, 0.4, 1.8], [-0.8, 0.4, 1.8],
                    [-0.8, 0.4, 1.6], [-0.8, 0.4, 1.6], [-0.8, 0.4, 1.8], [-0.8, 0.4, 1.8], [-0.8, 0.4, 1.6], [-0.8, 0.4, 1.6]]
target_orientations = [p.getQuaternionFromEuler([0, 0, 0]) for _ in target_positions]

#trailDuration is duration (in seconds) after debug lines will be removed automatically
#use 0 for no-removal
trailDuration = 15.
basepos = [0, 0, 0]
ang = 0
ang = 0

def move_to_positions(target_positions, threshold=0.001, maxIter=100):
    for target in target_positions:
        closeEnough = False
        iter = 0
        while not closeEnough and iter < maxIter:
            jointPoses = p.calculateInverseKinematics(hiwinId, hiwinEndEffectorIndex, target)
            for i, jointPose in enumerate(jointPoses):
                p.setJointMotorControl2(bodyIndex=hiwinId,
                                        jointIndex=i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=jointPose,
                                        targetVelocity=0,
                                        force=500,
                                        positionGain=1,
                                        velocityGain=0.1)
            # Check if the end effector is close enough to the target
            ls = p.getLinkState(hiwinId, hiwinEndEffectorIndex)
            newPos = ls[4]  # World position of the end-effector
            diff = [target[0] - newPos[0], target[1] - newPos[1], target[2] - newPos[2]]
            dist2 = diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2
            closeEnough = dist2 < threshold
            iter += 1
            if useSimulation:
                p.stepSimulation()
            time.sleep(0.1)  # Small delay to allow for simulation to catch up

def accurateCalculateInverseKinematics(hiwinId, endEffectorId, targetPos, threshold, maxIter):
  closeEnough = False
  iter = 0
  dist2 = 1e30
  while (not closeEnough and iter < maxIter):
    jointPoses = p.calculateInverseKinematics(hiwinId, hiwinEndEffectorIndex, targetPos)
    print("IK solution length:", len(jointPoses))
    for i in range(len(jointPoses)):
      p.resetJointState(hiwinId, i, jointPoses[i])

    ls = p.getLinkState(hiwinId, hiwinEndEffectorIndex)
    newPos = ls[5]
    diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
    dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
    closeEnough = (dist2 < threshold)
    iter = iter + 1
  #print ("Num iter: "+str(iter) + "threshold: "+str(dist2))
  return jointPoses

wheels = [2, 3, 4, 5]
#(2, b'front_left_wheel', 0, 7, 6, 1, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'front_left_wheel_link')
#(3, b'front_right_wheel', 0, 8, 7, 1, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'front_right_wheel_link')
#(4, b'rear_left_wheel', 0, 9, 8, 1, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'rear_left_wheel_link')
#(5, b'rear_right_wheel', 0, 10, 9, 1, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'rear_right_wheel_link')
wheelVelocities = [0, 0, 0, 0]
wheelDeltasTurn = [1, -1, 1, -1]
wheelDeltasFwd = [1, 1, 1, 1]
# Capture the initial position of the end-effector
link_state = p.getLinkState(hiwinId, hiwinEndEffectorIndex)
initial_position = link_state[4]

# Initialize variable for previous position
previous_position = initial_position

# Loop through each target position and orientation
for pos, ori in zip(target_positions, target_orientations):
    jointPositions = p.calculateInverseKinematics(hiwinId, hiwinEndEffectorIndex, pos, ori)
    
    for jointIndex, jointPosition in enumerate(jointPositions):
        # Increase targetVelocity and force for faster and stronger joint movements
        p.setJointMotorControl2(hiwinId, jointIndex, p.POSITION_CONTROL, targetPosition=jointPosition,
                                targetVelocity=1.0,  # Adjust as needed
                                force=500,  # Adjust as needed
                                positionGain=0.03,  # Optional: Adjust PID parameters for faster response
                                velocityGain=1.0)  # Optional: Adjust PID parameters for faster response
        
    
    # Reduce the number of simulation steps if desired for quicker movements
    for _ in range(100): 
        # 100 steps per second
        p.stepSimulation()
        time.sleep(1./480.)  # Simulation timestep

        # Get the end-effector's current position
        link_state = p.getLinkState(hiwinId, hiwinEndEffectorIndex)
        current_position = link_state[4]

        # Draw a line between the previous and current positions of the end-effector
        p.addUserDebugLine(previous_position, current_position, [1, 0, 0], 1, trailDuration)

        # Update the previous position
        previous_position = current_position