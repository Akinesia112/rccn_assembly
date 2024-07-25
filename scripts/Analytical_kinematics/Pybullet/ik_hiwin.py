import pybullet as p
import pybullet_data
import time
import math

# Start PyBullet in graphical mode
p.connect(p.GUI)

# Load the brick json file into pybullet
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

# Load the robot URDF model
p.setAdditionalSearchPath(pybullet_data.getDataPath())
#robotId = p.loadURDF("hiwin_ra620_1621_support/urdf/ra620_1621.urdf", [0, 0, 0], useFixedBase=1)
urdf_filename = r"D:\RCCN_Assembly\rccn_assembly\scripts\Data\hiwin_ra610_1476_support\urdf\ra610_1476.urdf"
# urdf_filename = r"C:\Users\Acer\Downloads\OmniIsaacGymEnvs-HiwinReacher-main (1)\OmniIsaacGymEnvs-HiwinReacher-main\thirdparty\hiwin_ra620_1621_support\urdf\ra620_1621.urdf"
robotId = p.loadURDF(urdf_filename, [0, 0, 0], useFixedBase=1)

'''
robotEndEffectorIndex = 6
numJoints = p.getNumJoints(robotId)
if (numJoints != 7):
  exit()

#lower limits for null space
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
#upper limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
#joint ranges for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
#restposes for null space
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
#joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

for i in range(numJoints):
  p.resetJointState(robotId, i, rp[i])

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
'''
# Set gravity (optional, may be required depending on the simulation)
p.setGravity(0, 0, -9.81)

# Define the target positions and orientations for the robotic arm's end-effector
# target_positions = [[-1.6, 0, -1.2], [-1.6, -0.8, -1.2], [-1.6, -0.8, -2.0]]
target_positions = [[0.8, 0, 0.6], [0.8, 0.4, 0.6], [0.8, 1.6, 1.0], [1.6, 1.6, 1.0], [1.6, 0.8, 1.0], 
                    [1.6, 0.8, 1.2], [1.6, 0, 1.2],[0.8, 0, 1.2], [0.8, 0, 0.6],[1.6, 1.6, 1.6], [1.6, 0.8, 1.6],
                    [1.6, 0.8, 1.8], [1.6, 0, 1.8], [0.8, 0, 1.8], [0.8, 0, 1.6], [0.8, 0.4, 1.6], [0.8, 0.4, 1.8],
                    [-0.8, -1.6, 1.8], [-1.6, -1.6, 1.8], [-1.6, -0.8, 1.8], [-1.6, -0.8, 1.6], [-1.6, 0, 1.6], [-0.8, 0, 1.6],
                    [-0.8, 0, 1.8], [-0.8, 0.4, 1.8], [-0.8, 0.4, 1.6], [-0.8, 0.4, 1.6], [-0.8, 0.4, 1.8], [-0.8, 0.4, 1.8],
                    [-0.8, 0.4, 1.6], [-0.8, 0.4, 1.6], [-0.8, 0.4, 1.8], [-0.8, 0.4, 1.8], [-0.8, 0.4, 1.6], [-0.8, 0.4, 1.6]]

target_orientations = [p.getQuaternionFromEuler([0, 0, 0]) for _ in target_positions]

# Variable to store the initial position of the end-effector
initial_position = None
robotEndEffectorIndex = 6
trailDuration = 15

# Capture the initial position of the end-effector
link_state = p.getLinkState(robotId, robotEndEffectorIndex)
initial_position = link_state[4]

# Initialize variable for previous position
previous_position = initial_position

# Loop through each target position and orientation
for pos, ori in zip(target_positions, target_orientations):
    # Calculate the inverse kinematics for the current target position and orientation
    jointPositions = p.calculateInverseKinematics(robotId, robotEndEffectorIndex, pos, ori)

    # Apply the calculated joint positions to the robot
    for jointIndex, jointPosition in enumerate(jointPositions):
        p.setJointMotorControl2(robotId, jointIndex, p.POSITION_CONTROL, jointPosition)

    # Step through the simulation to update the robot's state
    for _ in range(100): 
        # 100 steps per second
        p.stepSimulation()
        time.sleep(1./240.)  # Simulation timestep

        # Get the end-effector's current position
        link_state = p.getLinkState(robotId, robotEndEffectorIndex)
        current_position = link_state[4]

        # Draw a line between the previous and current positions of the end-effector
        p.addUserDebugLine(previous_position, current_position, [1, 0, 0], 1, trailDuration)

        # Update the previous position
        previous_position = current_position

