import networkx as nx
import pathlib
import time
from compas_assembly.datastructures import Assembly
from compas_view2.app import App
from compas.geometry import Point, Polyline

# 载入组装
CWD = pathlib.Path(__file__).parent.absolute()
FILE = CWD.parent.parent / "scripts" / "output" / "assembly_interface_from_rhino_test_2.json"
assembly = Assembly.from_json(FILE)

# BFS
undirected_graph = assembly.graph.to_networkx().to_undirected()
node_coordinates = []

# 准备开始 BFS 遍历
bfs_queue = []
visited = set()

# 找到第一个起点
for cc in nx.connected_components(undirected_graph):
    start_node = next(iter(cc))
    bfs_queue.append(start_node)
    visited.add(start_node)
    break

# 进行 BFS 遍历并收集坐标
while bfs_queue:
    current_node = bfs_queue.pop(0)
    x = assembly.graph.node_attribute(current_node, 'x')
    y = assembly.graph.node_attribute(current_node, 'y')
    z = assembly.graph.node_attribute(current_node, 'z')
    node_coordinates.append([x, y, z])  # Collect the coordinates as lists

    for neighbor in undirected_graph.neighbors(current_node):
        if neighbor not in visited:
            bfs_queue.append(neighbor)
            visited.add(neighbor)

# Initialize the viewer
viewer = App(viewmode="shaded", enable_sidebar=True, width=1600, height=900)

# Add nodes as points and create a polyline from the BFS node coordinates
for coord in node_coordinates:
    viewer.add(Point(*coord), pointsize=10, pointcolor=(1, 0, 0))

curveobj = viewer.add(Polyline(node_coordinates), linewidth=2)

@viewer.button(text="Reset View")
def click():
    viewer.view.reset()

viewer.view.camera.scale = 1000
viewer.view.camera.position = [3000, 3000, 3000]
viewer.view.camera.distance = 300

viewer.show()
