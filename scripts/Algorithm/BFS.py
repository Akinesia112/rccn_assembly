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

CWD = pathlib.Path(__file__).parent.absolute()
FILE = CWD.parent.parent / "scripts" / "output" / "assembly_interface_from_rhino_demo.json"
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
        viewer.add(interface.polygon, facecolor=(0.0,0,0.0))

for edge_interface in assembly.interfaces():
    viewer.add(edge_interface.polygon, facecolor=(0.0,0,0.0))
'''
# BFS
# 將有向圖轉換為無向圖
undirected_graph = assembly.graph.to_networkx().to_undirected()

# 初始化節點排序計數器
node_count = 0

# 找到所有連通分量
for cc in nx.connected_components(undirected_graph):
    # 選擇每個連通分量的一個起始節點
    start_node = next(iter(cc))
    
    # 對該連通分量進行 BFS
    bfs_tree = nx.bfs_tree(undirected_graph, source=start_node)
    sorted_nodes = list(bfs_tree)

    # 標註 BFS 樹中的節點
    for node in sorted_nodes:
        coordinate = assembly.graph.node_attributes(node, ['x', 'y', 'z'])
        if coordinate is not None:
            t = Text("{}".format(node_count), [coordinate[0],coordinate[1],coordinate[2]], height=25)
            viewer.add(t)
            node_count += 1

network = Network.from_networkx(assembly.graph.to_networkx())

print(network.summary())
viewer.add(network)
viewer.view.camera.scale = 2000
#viewer.add(Collection(polygons), facecolor=(0.3,0.3,0.3))
viewer.view.camera.position = [1000, 1000, 1000]
viewer.show()