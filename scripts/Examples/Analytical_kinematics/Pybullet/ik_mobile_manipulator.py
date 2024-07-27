import pybullet as p
import time
import pybullet_data
import math

# Initialize PyBullet
clid = p.connect(p.SHARED_MEMORY)
#clid = p.connect(p.GUI)

if (clid < 0):
    p.connect(p.GUI)

p.setPhysicsEngineParameter(enableConeFriction=0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set smaller time step for simulation
p.setTimeStep(1./240.)  # Smaller time steps for finer simulation

# Load the environment plane and URDFs
planeId = p.loadURDF("plane.urdf", [0, 0, -0.3])
husky = p.loadURDF("husky/husky.urdf", [0, 0, -0.310270],
                   [0.00, -0.000, 0, 0.083659], globalScaling = 1.5)
robot_urdf_path = r"D:\RCCN_Assembly\rccn_assembly\scripts\Data\hiwin_ra610_1476_support\urdf\ra610_1476.urdf"  # Update this path
robotId = p.loadURDF(robot_urdf_path, [0, 0, 0.3], useFixedBase=1)
jointPositions = [3.559609, 0.411182, 0.862129, 1.744441, 0.077299, 3.559609, 0.0, 0.0, 0.0]
for jointIndex in range(p.getNumJoints(robotId)):
    p.resetJointState(robotId, jointIndex, jointPositions[jointIndex])


#put hiwin on top of husky

cid = p.createConstraint(husky, -1, robotId, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0., 0., -.5],
                         [0, 0, 0, 1])

baseorn = p.getQuaternionFromEuler([3.1415, 0, 0.3])
baseorn = [0, 0, 0, 1]
#[0, 0, 0.707, 0.707]

#p.resetBasePositionAndOrientation(hiwinId,[0,0,0],baseorn)#[0,0,0,1])
numJoints = p.getNumJoints(robotId)
# Optionally, create a fixed joint between husky and the robot if needed
# p.createConstraint(husky, -1, robotId, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])

# Simulation setup
p.setGravity(0, 0, -9.81)
robotEndEffectorIndex = 6  # Update this based on your robot's end effector
trailDuration = 20

# Define target positions and orientations
target_positions = [[0.8, 0, 0.6], [0.8, 0.4, 0.6], [0.8, 1.6, 1.0], [1.6, 1.6, 1.0], [1.6, 0.8, 1.0], 
                    [1.6, 0.8, 1.2], [1.6, 0, 1.2],[0.8, 0, 1.2], [0.8, 0, 0.6],[1.6, 1.6, 1.6], [1.6, 0.8, 1.6],
                    [1.6, 0.8, 1.8], [1.6, 0, 1.8], [0.8, 0, 1.8], [0.8, 0, 1.6], [0.8, 0.4, 1.6], [0.8, 0.4, 1.8],
                    [-0.8, -1.6, 1.8], [-1.6, -1.6, 1.8], [-1.6, -0.8, 1.8], [-1.6, -0.8, 1.6], [-1.6, 0, 1.6], [-0.8, 0, 1.6],
                    [-0.8, 0, 1.8], [-0.8, 0.4, 1.8], [-0.8, 0.4, 1.6], [-0.8, 0.4, 1.6], [-0.8, 0.4, 1.8], [-0.8, 0.4, 1.8],
                    [-0.8, 0.4, 1.6], [-0.8, 0.4, 1.6], [-0.8, 0.4, 1.8], [-0.8, 0.4, 1.8], [-0.8, 0.4, 1.6], [-0.8, 0.4, 1.6]]
target_orientations = [p.getQuaternionFromEuler([0, 0, 0]) for _ in target_positions]


prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
useNullSpace = 0

useOrientation = 0
#If we set useSimulation=0, it sets the arm pose to be the IK result directly without using dynamic control.
#This can be used to test the IK result accuracy.
useSimulation = 1
useRealTimeSimulation = 1
p.setRealTimeSimulation(useRealTimeSimulation)
#trailDuration is duration (in seconds) after debug lines will be removed automatically
#use 0 for no-removal
basepos = [0, 0, 0]
ang = 0


def accurateCalculateInverseKinematics(robotId, endEffectorId, targetPos, threshold, maxIter):
  closeEnough = False
  iter = 0
  dist2 = 1e30
  while (not closeEnough and iter < maxIter):
    jointPoses = p.calculateInverseKinematics(robotId, robotEndEffectorIndex, targetPos)
    for i in range(numJoints):
      p.resetJointState(robotId, i, jointPoses[i])
    ls = p.getLinkState(robotId, robotEndEffectorIndex)
    newPos = ls[4]
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

# Visualization and movement
previous_position = None  # Initialize
for pos, ori in zip(target_positions, target_orientations):
    jointPositions = p.calculateInverseKinematics(robotId, robotEndEffectorIndex, pos, ori)
    for jointIndex, jointPosition in enumerate(jointPositions):
        p.setJointMotorControl2(robotId, jointIndex, p.POSITION_CONTROL, jointPosition, force=500)
    
    for _ in range(100):
        p.stepSimulation()
        time.sleep(1./500.)
        if previous_position is not None:
            current_position = p.getLinkState(robotId, robotEndEffectorIndex)[0]
            p.addUserDebugLine(previous_position, current_position, [1, 0, 0], 1, trailDuration)
        previous_position = p.getLinkState(robotId, robotEndEffectorIndex)[0]
