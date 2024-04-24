import compas_fab
from compas_fab.backends import PyBulletClient
from compas_fab.robots import Configuration
import pathlib
import networkx as nx
from compas_assembly.datastructures import Assembly
from compas.datastructures import Network
import pathlib
from compas_assembly.algorithms import assembly_interfaces_numpy
from compas_rbe.equilibrium import compute_interface_forces_cvx
from compas_view2.app import App
from compas_view2.shapes import Text
from compas_view2.objects import Object, NetworkObject, LineObject, PointObject
from compas.geometry import Point, Line, Frame, add_vectors, Sphere
from compas.colors import Color
from compas.geometry import Vector
from compas.geometry import centroid_points_weighted

# Assume function to check if the base position of the robot is within a specified range
def is_within_base_range(config, base_range):
    max_reach = 0.85  # The maximum reach of the robot is 0.85m
    for joint_value in config.joint_values:
        if abs(joint_value) > max_reach:
            return False
    return True

# load and process assembly function
def load_and_process_assembly(file_path):
    # Load the assembly from JSON
    assembly = Assembly.from_json(file_path)
    # Compute the assembly interfaces
    assembly_interfaces_numpy(assembly, nmax=10, amin=0.0001)

    # Compute the equilibrium geometry
    compute_interface_forces_cvx(assembly, solver='CPLEX', verbose=True)

    return assembly

# collect the centroid of all nodes
def collect_node_positions(assembly):
    network = Network()
    positions = []  # To get the centroid of each block, we need to convert the block to a network
    for node in assembly.graph.nodes():
        block = assembly.graph.node_attribute(node, 'block')
        centroid = block.centroid()
        network.add_node(node, x=centroid[0], y=centroid[1], z=centroid[2])
        positions.append([centroid[0], centroid[1], centroid[2]])  # 將 Point 轉換為列表並添加到位置列表中
    return positions

def normalize_positions(positions):
    # Extract x, y, z coordinates
    x_vals = [pos[0] for pos in positions]
    y_vals = [pos[1] for pos in positions]
    z_vals = [pos[2] for pos in positions]

    # Compute the min and max values for each dimension
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    min_z, max_z = min(z_vals), max(z_vals)

    # Regularize the positions
    normalized_positions = []
    for x, y, z in positions:
        norm_x = round((x - min_x) / (max_x - min_x) if max_x != min_x else 0.5, 3)
        norm_y = round((y - min_y) / (max_y - min_y) if max_y != min_y else 0.5, 3)
        norm_z = round((z - min_z) / (max_z - min_z) if max_z != min_z else 0.5, 3)
        normalized_positions.append([norm_x, norm_y, norm_z])

    return normalized_positions

# Path of the assembly file
CWD = pathlib.Path(__file__).parent.absolute()
test_2_file = CWD.parent.parent / "output" / "assembly_interface_from_rhino_test_2.json"

# Load and process the assembly
assembly_test_2 = load_and_process_assembly(test_2_file)
node_positions = collect_node_positions(assembly_test_2)

# utilize this function to normalize your position data
normalized_positions = normalize_positions(node_positions)

# define the end effector position as the position of all nodes of the assembly
frames = [Frame(position, [1, 0, 0], [0, 1, 0]) for position in normalized_positions]

# Load the URDF file
# urdf_filename = compas_fab.get('universal_robot/ur_description/urdf/ur5.urdf') 
# ---------------------------------------------------------------
# compas_fab.get() is a function to get the path of a file in the environment, such as:
# D:\anaconda3\envs\rccn_assembly\Lib\site-packages\compas_fab\data\universal_robot\ur_description\urdf\ur5.urdf
# ---------------------------------------------------------------
#urdf_filename =  r'D:\RCCN_Assembly\rccn_assembly\hiwin_ra620_1621_support\urdf\ra620_1621.urdf'
urdf_filename = r"C:\Users\Acer\Downloads\urdf_model-main\urdf_model-main\hiwin_robot_arm\combine_description\hiwin_arm\Hiwin_RA610_1476_GC.urdf"
# use PyBullet to compute the inverse kinematics
with PyBulletClient(connection_type='direct') as client:
    robot = client.load_robot(urdf_filename)
    start_configuration = robot.zero_configuration()
    options = dict(max_results=20, high_accuracy_threshold=1e-6, high_accuracy_max_iter=20)
    
    i = 0
    for frame in frames:
        i += 1
        print("==================================== i = ", i, "====================================")
        print(f"Finding solutions for frame: {frame}")
        for config in robot.iter_inverse_kinematics(frame, start_configuration, options=options):
                print("Found Configuration:", config)
                print("......................................................................................")