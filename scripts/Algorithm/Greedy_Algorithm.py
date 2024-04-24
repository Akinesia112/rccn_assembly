import networkx as nx
from compas_assembly.datastructures import Assembly
from compas_view2.app import App
import pathlib

# 载入组装
CWD = pathlib.Path(__file__).parent.absolute()
FILE = CWD.parent / "output" / "assembly_interface_from_rhino_curve.json"
assembly = Assembly.from_json(FILE)

# 初始化视图器
viewer = App(width=2000, height=1000)

# 转换为无向图
undirected_graph = assembly.graph.to_networkx().to_undirected()

# 初始化已访问节点集合
visited = set()

# 用于存储已添加的砖块对象
added_blocks = {}

# 贪婪算法寻找砖块放置序列
def find_greedy_path(start_node):
    path = []
    current_node = start_node
    while current_node is not None:
        path.append(current_node)
        visited.add(current_node)
        min_distance = float('inf')
        next_node = None

        # 找到距离最近的未访问邻居
        for neighbor in undirected_graph.neighbors(current_node):
            if neighbor not in visited:
                distance = nx.shortest_path_length(undirected_graph, current_node, neighbor, weight='weight')
                if distance < min_distance:
                    min_distance = distance
                    next_node = neighbor

        # 如果找不到未访问的邻居，尝试从未访问的节点中选择一个新的起点
        if next_node is None:
            unvisited_nodes = set(undirected_graph.nodes()) - visited
            if unvisited_nodes:
                next_node = unvisited_nodes.pop()

        current_node = next_node
    return path

# 对整个图应用贪婪算法
full_path = find_greedy_path(next(iter(undirected_graph.nodes())))

# 定义更新视图的函数
@viewer.on(interval=10)
def update_view(frame):
    if full_path:
        current_node = full_path.pop(0)
        block = assembly.graph.node_attribute(current_node, 'block')

        if current_node not in added_blocks:
            obj = viewer.add(block, opacity=0.5)
            added_blocks[current_node] = obj
    else:
        viewer.stop()

# 设置视图器的相机位置和缩放
viewer.view.camera.scale = 1000
viewer.view.camera.position = [3000, 3000, 3000]
viewer.view.camera.distance = 8000

# 运行视图器
viewer.run()
