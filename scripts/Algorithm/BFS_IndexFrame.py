import networkx as nx
import csv
import pathlib
from compas_assembly.datastructures import Assembly
from compas.geometry import Frame

# 载入组装
CWD = pathlib.Path(__file__).parent.absolute()
FILE = CWD.parent.parent / "scripts" / "output" / "assembly_interface_from_rhino_test_2.json"
assembly = Assembly.from_json(FILE)

# BFS
undirected_graph = assembly.graph.to_networkx().to_undirected()
node_data = []

# 准备开始 BFS 遍历
bfs_queue = []
visited = set()

# 找到第一个起点
for cc in nx.connected_components(undirected_graph):
    start_node = next(iter(cc))
    bfs_queue.append(start_node)
    visited.add(start_node)
    break

# 进行 BFS 遍历并收集数据
while bfs_queue:
    current_node = bfs_queue.pop(0)
    x = assembly.graph.node_attribute(current_node, 'x')
    y = assembly.graph.node_attribute(current_node, 'y')
    z = assembly.graph.node_attribute(current_node, 'z')
    # Extract frame data
    for neighbor in undirected_graph.neighbors(current_node):
        if neighbor not in visited:
            visited.add(neighbor)
            bfs_queue.append(neighbor)
            if (current_node, neighbor) in undirected_graph.edges:
                print("Current Node:", current_node, "Neighbor:", neighbor)
                edge = (current_node, neighbor)
                interfaces = assembly.edge_interfaces(edge)
                for interface in interfaces:
                    print("Interface:", interface)
                    frame = interface.frame
                    print("frame:", frame)
                    point = frame.point
                    print("Point:", point)
                    xaxis = frame.xaxis
                    print("Xaxis:", xaxis)
                    yaxis = frame.yaxis
                    print("Yaxis:", yaxis)
                node_data.append([current_node, neighbor, x, y, z, point, xaxis, yaxis])

# 将结果写入 CSV 文件
output_path = pathlib.Path(CWD, "csv","bfs_coordinates_with_frame_data.csv")
with open(output_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Node', 'X', 'Y', 'Z', 'Point', 'Xaxis', 'Yaxis'])
    writer.writerows(node_data)
