import compas
from compas.robots import LocalPackageMeshLoader
from compas.robots import RobotModel

import compas_fab

'''
Loading model from disk
The installation of COMPAS FAB includes some robot models which are used to exemplify loading from disk:
'''
# Set high precision to import meshes defined in meters
compas.PRECISION = '12f'

# Locate the URDF file inside compas fab installation
urdf = compas_fab.get('universal_robot/ur_description/urdf/ur5.urdf')

# Create robot model from URDF
model = RobotModel.from_urdf_file(urdf)

# Also load geometry
loader = LocalPackageMeshLoader(compas_fab.get('universal_robot'), 'ur_description')
model.load_geometry(loader)

print(model)

'''
Loading model from Github
Since a large amount of robot models defined in URDF are available on Github, 
COMPAS FAB provides a specialized loader that follows the conventions defined 
by ROS to locate a Robot’s model and geometry files.
'''
import compas
from compas.robots import GithubPackageMeshLoader
from compas.robots import RobotModel

# Set high precision to import meshes defined in meters
compas.PRECISION = '12f'

# Select Github repository, package and branch where the model is stored
repository = 'ros-industrial/abb'
package = 'abb_irb6600_support'
branch = 'kinetic-devel'

github = GithubPackageMeshLoader(repository, package, branch)
urdf = github.load_urdf('irb6640.urdf')

# Create robot model from URDF
model = RobotModel.from_urdf_file(urdf)

# Also load geometry
model.load_geometry(github)

print(model)

'''
Loading model from ROS
Note:
The following example uses the ROS backend and loads the robot description model from it. 
Before running it, please make sure you have the ROS backend correctly configured and the Panda Demo started.
In most situations, we will load the robot model directly from a running ROS instance. 
The following code exemplifies how to do that.
'''

import compas
from compas_fab.backends import RosClient

# Set high precision to import meshes defined in meters
compas.PRECISION = '12f'

with RosClient() as ros:
    robot = ros.load_robot(load_geometry=True)

    print(robot.model)

'''
Additionally, the ROS loader allows to cache the results locally for faster reloads, 
to enable this behavior, pass an argument with the folder where the cache should be stored:
'''

import os

import compas
from compas_fab.backends import RosClient

# Set high precision to import meshes defined in meters
compas.PRECISION = '12f'

with RosClient() as ros:
    # Load complete model from ROS and set a local cache location
    local_directory = os.path.join(os.path.expanduser('~'), 'robot_description', 'robot_name')
    robot = ros.load_robot(load_geometry=True, local_cache_directory=local_directory)

    print(robot.model)

'''
Visualizing robot models
Once a model is loaded, we can visualize it in our favorite design environment.

COMPAS includes the concept of artists: classes that assist with the visualization of datastructures and models, 
in a way that maintains the data separated from the specific CAD interfaces, 
while providing a way to leverage native performance of the CAD environment.

In the main library there are artists for various datastructures (meshes, networks, etc), 
including a RobotModelArtist to visualize robots. Robot artists allow visualizing robot models easily and efficiently.

The following example illustrates how to load an entire robot model from ROS and render it in Rhino:
'''
import compas
from compas_fab.backends import RosClient
from compas_rhino.artists import RobotModelArtist

# Set high precision to import meshes defined in meters
compas.PRECISION = '12f'

with RosClient() as ros:
    # Load complete model from ROS
    robot = ros.load_robot(load_geometry=True)

    # Visualize robot
    robot.artist = RobotModelArtist(robot.model, layer='COMPAS FAB::Example')
    robot.artist.clear_layer()
    robot.artist.draw_visual()