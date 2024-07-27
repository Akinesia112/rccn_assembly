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

def is_directed(assembly):
    """
    Determines if the graph is directed based on the NetworkX graph representation.
    """
    nx_graph = assembly.graph.to_networkx() # This line means that the graph is converted to a NetworkX graph
    return nx_graph.is_directed()

def is_weighted(assembly):
    """
    Determines if the graph is weighted, checking for a specific attribute that contains mechanical information.
    """
    nx_graph = assembly.graph.to_networkx()
    for _, _, attr in nx_graph.edges(data=True):
        # Replace 'forces' with the actual attribute name
        if 'forces' in attr:  
            return True
    return False

# Visualize the assembly
def load_and_process_assembly(file_path):
    # Load the assembly from JSON
    assembly = Assembly.from_json(file_path)
    # Compute the assembly interfaces
    assembly_interfaces_numpy(assembly, nmax=10, amin=0.0001)

    # Compute the equilibrium geometry
    compute_interface_forces_cvx(assembly, solver='CPLEX', verbose=True)

    return assembly

'''
# Forces Computing Functions.
# From compas_assembly/src/compas_assembly/datastructures/interface.py
# https://github.com/BlockResearchGroup/compas_assembly/blob/main/src/compas_assembly/datastructures/interface.py#L9

def contactforces(self):
    lines = []
    if not self.forces:
        return lines
    frame = self.frame
    w = frame.zaxis
    for point, force in zip(self.points, self.forces):
        point = Point(*point)
        force = force["c_np"] - force["c_nn"]
        p1 = point + w * force * 0.5
        p2 = point - w * force * 0.5
        lines.append(Line(p1, p2))
    return lines

def compressionforces(self):
    lines = []
    if not self.forces:
        return lines
    frame = self.frame
    w = frame.zaxis
    for point, force in zip(self.points, self.forces):
        point = Point(*point)
        force = force["c_np"] - force["c_nn"]
        if force > 0:
            p1 = point + w * force * 0.5
            p2 = point - w * force * 0.5
            lines.append(Line(p1, p2))
    return lines

def tensionforces(self):
    lines = []
    if not self.forces:
        return lines
    frame = self.frame
    w = frame.zaxis
    for point, force in zip(self.points, self.forces):
        point = Point(*point)
        force = force["c_np"] - force["c_nn"]
        if force < 0:
            p1 = point + w * force * 0.5
            p2 = point - w * force * 0.5
            lines.append(Line(p1, p2))
    return lines

def frictionforces(self):
    lines = []
    if not self.forces:
        return lines
    frame = self.frame
    u, v = frame.xaxis, frame.yaxis
    for point, force in zip(self.points, self.forces):
        point = Point(*point)
        ft_uv = (u * force["c_u"] + v * force["c_v"]) * 0.5
        p1 = point + ft_uv
        p2 = point - ft_uv
        lines.append(Line(p1, p2))
    return lines

def resultantforce(self):
    if not self.forces:
            return []
    frame = self.frame
    w, u, v = frame.zaxis, frame.xaxis, frame.yaxis
    normalcomponents = [f["c_np"] - f["c_nn"] for f in self.forces]
    sum_n = sum(normalcomponents)
    sum_u = sum(f["c_u"] for f in self.forces)
    sum_v = sum(f["c_v"] for f in self.forces)
    position = Point(*centroid_points_weighted(self.points, normalcomponents))
    forcevector = (w * sum_n + u * sum_u + v * sum_v) * 0.5
    p1 = position + forcevector
    p2 = position - forcevector
    return [Line(p1, p2)]
'''

def visualize_forces(interface, viewer):
    frame = interface.frame
    w, u, v = frame.zaxis, frame.xaxis, frame.yaxis

    # Visualize contact forces
    #for line in interface.contactforces:
        #viewer.add(line, linecolor=(0, 0, 0), linewidth=2)  # Black for contact forces

    # Visualize compression forces
    #for line in interface.compressionforces:
        #viewer.add(line, linecolor=(0.678, 0.847, 0.902), linewidth=2) # Light Blue (r, g, b), (173, 216, 230) for compression forces

    # Visualize tension forces
    #for line in interface.tensionforces:
        #viewer.add(line, linecolor=(1.0, 0, 0), linewidth=2)  # Red for tension forces

    # Visualize friction forces
    #for line in interface.frictionforces:
        #viewer.add(line, linecolor=(0.0, 1.0, 0.0), linewidth=2)  # Green for friction forces

    # Visualize resultant forces
    #for line in interface.resultantforce:
        #viewer.add(line, linecolor=(1.0, 0, 1.0), linewidth=2) # Pink for resultant forces
    
    
# Visualize the assembly
def visualize_assembly(assembly):
    viewer = App()
    assembly.interfaces()
    # Create a network from the assembly
    network = Network()

    # Add nodes (blocks) to the network
    for node in assembly.graph.nodes():
        block = assembly.graph.node_attribute(node, 'block')
        viewer.add(block, opacity=0.5)

        centroid = block.centroid()
        network.add_node(node, x=centroid[0], y=centroid[1], z=centroid[2])
        # print x, y, z
        print("X:", centroid[0], "Y:", centroid[1], "Z:", centroid[2])

    for edge in assembly.graph.edges():
        u, v = edge
        network.add_edge(u, v)
        interface = assembly.graph.edge_attribute(edge, "interface")

    for interface in assembly.interfaces():
        viewer.add(interface.polygon, facecolor=(0,0,0), opacity=1)
        visualize_forces(interface, viewer)
        
        
    
    network = Network.from_networkx(assembly.graph.to_networkx())
    print(network.summary())
    viewer.add(network)

    viewer.view.camera.scale = 1000
    viewer.view.camera.position = [3000, 3000, 3000]
    viewer.view.camera.distance = 1000

    viewer.run()
    
# Paths to the JSON files
# Load the assembly
CWD = pathlib.Path(__file__).parent.absolute()
cwd = pathlib.Path(__file__).parent.absolute()
test_2_file = CWD.parent.parent / "scripts" / "output" / "assembly_interface_from_rhino_hexa.json"
collapse_file = CWD.parent.parent  / "scripts" / "output" / "assembly_interface_from_rhino_collapse.json"

'''
# Load the assembly from JSON by forloop
for file in CWD.parent.glob('output/*.json'):
    print("File:", file)
    assembly = Assembly.from_json(file)
    print(assembly)
    # Check if the graph is directed and weighted
    directed = is_directed(assembly)
    weighted = is_weighted(assembly)
    print("Directed:", directed)
    print("Weighted:", weighted)
'''
# Load and process assemblies
assembly_test_2 = load_and_process_assembly(test_2_file)
assembly_collapse = load_and_process_assembly(collapse_file)

# Visualize assemblies
visualize_assembly(assembly_test_2)
visualize_assembly(assembly_collapse)
