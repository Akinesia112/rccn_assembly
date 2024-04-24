import networkx as nx
import matplotlib.pyplot as plt
import pathlib
from compas_assembly.datastructures import Assembly
from compas.datastructures import Network
from compas_assembly.viewer.app import DEMViewer
from compas_view2.app import App
from compas_view2.objects import Collection
from compas.geometry import Scale
import json
import pathlib
from compas_assembly.datastructures import Block
import compas.datastructures as cd
from compas_view2.shapes import Text

# 讀取JSON文件
with open(r'D:\RCCN_Assembly\rccn_assembly\output\assembly_interface_from_rhino_door.json') as f:
    data = json.load(f)


forces_data = {}  # 將 forces_data 從空列表改成空字典
for edge_key, edge_value in data["graph"]["edge"].items():
    if "interfaces" in edge_value:
        interface = edge_value["interfaces"][0]["value"]  
        if "forces" in interface:
            forces_data[edge_key] = interface["forces"]

# 提取所有 vertex 
vertex_data = {}
for node_key, node_value in data["graph"]["node"].items():
    if "block" in node_value:
        block = node_value["block"]["value"]
        if "vertex" in block:
            vertex_data[node_key] = block["vertex"]


# 结果
print("Forces data:")
for edge_key, forces in forces_data.items():
    print(f'Edge {edge_key}:')
    for force in forces:
        print(f'c_nn: {force["c_nn"]}, c_np: {force["c_np"]}, c_u: {force["c_u"]}, c_v: {force["c_v"]}')


print("\nVertex data:")
for node_key, vertices in vertex_data.items():
    for vertex_key, vertex_value in vertices.items():
        print(f'Node {node_key}, Vertex {vertex_key}: x: {vertex_value["x"]}, y: {vertex_value["y"]}, z: {vertex_value["z"]}')

# 創建一個無向圖
G = nx.Graph()

# 添加節點
for node_key, node_value in data["graph"]["node"].items():
    G.add_node(node_key)
'''
# 添加邊
for edge_key, edge_value in data["graph"]["edge"].items():
    node1, node2 = edge_key, list(edge_value["interfaces"].keys())[0]
    G.add_edge(node1, node2)
'''
# 定義排序條件的比較函數
def compare_nodes(node1, node2):
    forces_data1 = node1.get('forces', [])
    forces_data2 = node2.get('forces', [])

    # 比较 Forces data
    for i in range(min(len(forces_data1), len(forces_data2))):
        if forces_data1[i]['c_nn'] != forces_data2[i]['c_nn']:
            return forces_data1[i]['c_nn'] - forces_data2[i]['c_nn']
        if forces_data1[i]['c_np'] != forces_data2[i]['c_np']:
            return forces_data1[i]['c_np'] - forces_data2[i]['c_np']
        if forces_data1[i]['c_u'] != forces_data2[i]['c_u']:
            return forces_data1[i]['c_u'] - forces_data2[i]['c_u']
        if forces_data1[i]['c_v'] != forces_data2[i]['c_v']:
            return forces_data1[i]['c_v'] - forces_data2[i]['c_v']

    # 如果 forces_data 長度不同，按照長度比较
    if len(forces_data1) != len(forces_data2):
        return len(forces_data1) - len(forces_data2)

    # 比较 Vertex data
    vertex_data1 = node1.get('vertex', {})
    vertex_data2 = node2.get('vertex', {})

    for node_key in vertex_data1:
        for vertex_key in vertex_data1[node_key]:
            if vertex_data1[node_key][vertex_key]['x'] != vertex_data2[node_key][vertex_key]['x']:
                return vertex_data1[node_key][vertex_key]['x'] - vertex_data2[node_key][vertex_key]['x']
            if vertex_data1[node_key][vertex_key]['y'] != vertex_data2[node_key][vertex_key]['y']:
                return vertex_data1[node_key][vertex_key]['y'] - vertex_data2[node_key][vertex_key]['y']
            if vertex_data1[node_key][vertex_key]['z'] != vertex_data2[node_key][vertex_key]['z']:
                return vertex_data1[node_key][vertex_key]['z'] - vertex_data2[node_key][vertex_key]['z']

    # 如果 vertex_data 不同，按照長度比较
    if len(vertex_data1) != len(vertex_data2):
        return len(vertex_data1) - len(vertex_data2)

    return 0  # 如果两者相等


# 定義深度優先排序算法
def dfs_sort(graph, start_node, compare_func):
    visited = set()

    def dfs_recursive(node):
        if node not in visited:
            visited.add(node)
            neighbors = sorted(graph.neighbors(node), key=lambda n: compare_func(graph.nodes[node], graph.nodes[n]))
            for neighbor in neighbors:
                dfs_recursive(neighbor)

    dfs_recursive(start_node)
    return list(visited)

# 選擇一個起始節點
start_node = '0'
# 生成顏色映射
color_map = plt.cm.get_cmap('winter', len(G))
# color map 透明度
color_map._init()
color_map._lut[:, -1] = 0.5

# 使用深度優先排序算法疊完磚牆
sorted_nodes = dfs_sort(G, start_node, compare_nodes)


# 將顏色訊息添加到節點屬性
for node, color in zip(sorted_nodes, range(len(G))):
    color = color_map(color)

# 繪製圖形
nx.draw(G, with_labels=True, font_weight='bold', node_size=800, font_size=5, node_color= [color_map(i) for i in range(len(G))])

# 添加 colorbar
sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=0, vmax=len(G)-1))
sm.set_array([])  # 用於繪製 colorbar
plt.colorbar(sm, label='DFS Order')
plt.title("Graph Visualization")
plt.show()

CWD = pathlib.Path(__file__).parent.absolute()
FILE = CWD.parent / "output" / "assembly_interface_from_rhino_door.json"
assembly = Assembly.from_json(FILE)


polygons = []
for interface in assembly.interfaces():
    polygons.append(interface.polygon)

viewer = App()
#viewer = DEMViewer()
#viewer.add_assembly(assembly)

#assembly.interfaces()

for node in assembly.graph.nodes():
    block = assembly.graph.node_attribute(node, 'block')
    viewer.add(block, opacity=0.5)
'''
for edge in assembly.graph.edges():
    for interface in assembly.graph.edge_attribute(edge, "interfaces"):
        viewer.add(interface.polygon, facecolor=(0.3,0,0.3))

for edge_interface in assembly.interfaces():
    viewer.add(edge_interface.polygon, facecolor=(0.3,0,0.3))
'''
# DFS
sorted_nodes = list(nx.dfs_preorder_nodes(assembly.graph.to_networkx()))

# 計算最大深度
max_depth = 0
for node in sorted_nodes:
    depth = assembly.graph.node_attribute(node, 'depth')
    if depth is not None:
        max_depth = max(max_depth, depth)

# BFS Sort
sorted_nodes = list(nx.dfs_preorder_nodes(assembly.graph.to_networkx()))


# 標註節點之BFS順序
for i, node in enumerate(sorted_nodes):
    coordinate = assembly.graph.node_attributes(node, ['x', 'y', 'z'])  # 使用實際的節點屬性
    print(f'Node {node}: {coordinate}')
    if coordinate is not None:
        # Add tag
        print([coordinate[0],coordinate[1],coordinate[2]])
        t = Text("{}".format(i), [coordinate[0],coordinate[1],coordinate[2]], height=25)
        viewer.add(t)

network = Network.from_networkx(assembly.graph.to_networkx())

print(network.summary())
viewer.add(network)
viewer.view.camera.scale = 2000
#viewer.add(Collection(polygons), facecolor=(0.3,0.3,0.3))
viewer.view.camera.position = [1000, 1000, 1000]
viewer.show()