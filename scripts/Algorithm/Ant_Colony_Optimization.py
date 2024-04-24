import networkx as nx
import random
import math
import time
from compas_assembly.datastructures import Assembly
from compas_view2.app import App
import pathlib

# 载入组装
CWD = pathlib.Path(__file__).parent.absolute()
FILE = CWD.parent.parent / "output" / "assembly_interface_from_rhino_door.json"
assembly = Assembly.from_json(FILE)

# 初始化视图器
viewer = App(width=2000, height=1000)

# 转换为无向图
undirected_graph = assembly.graph.to_networkx().to_undirected()

# 用于存储已添加的砖块对象
added_blocks = {}

# 蚁群算法参数
num_ants = 100  # 蚂蚁数量
pheromone_evaporation = 0.1  # 信息素蒸发率
pheromone_deposit = 0.3  # 信息素沉积量
iterations = 50  # 迭代次数

# 初始化信息素
pheromones = {}
for edge in undirected_graph.edges():
    pheromones[edge] = 1.0
    pheromones[(edge[1], edge[0])] = 1.0  # 添加边的反方向

# 计算路径代价
def path_cost(graph, path):
    cost = 0
    for i in range(len(path) - 1):
        # 假设成本是路径中连续节点间的距离，可以根据实际情况调整
        if graph.has_edge(path[i], path[i + 1]):
            edge_data = graph.get_edge_data(path[i], path[i + 1])
            cost += edge_data.get('weight', 1)  # 如果没有提供权重，可以默认为1
        else:
            cost += float('inf')  # 如果路径不连续，返回无限大的成本
    return cost

# 蚁群算法主要函数
def ant_colony_optimization(graph):
    best_path = None
    best_cost = float('inf')


    iteration = 0
    while iteration < iterations:
        for _ in range(iterations):
            paths = [generate_path(graph) for _ in range(num_ants)]
            update_pheromones(paths)

            for path in paths:
                cost = path_cost(graph, path)
                if cost < best_cost:
                    best_path, best_cost = path, cost
        iteration += iterations
    # 如果找不到有效的路径，返回图中的随机路径
    if best_path is None:
        best_path = list(graph.nodes())

    return best_path

# 生成单个蚂蚁的路径
def generate_path(graph):
    path = []
    unvisited = set(graph.nodes())  # 创建一个包含所有未访问节点的集合
    current_node = random.choice(list(graph.nodes))
    path.append(current_node)
    unvisited.remove(current_node)

    while unvisited:
        next_node = choose_next_node(graph, current_node, unvisited)
        path.append(next_node)
        unvisited.remove(next_node)
        current_node = next_node

    return path

# 根据信息素选择下一个节点
def choose_next_node(graph, current_node, unvisited):
    neighbors = [n for n in graph.neighbors(current_node) if n in unvisited]
    if not neighbors:
        return random.choice(list(unvisited))  # 如果没有未访问的邻居，随机选择一个未访问的节点

    pheromone_levels = [pheromones[(current_node, n)] for n in neighbors]
    total_pheromone = sum(pheromone_levels)
    probabilities = [pheromone / total_pheromone for pheromone in pheromone_levels]

    return random.choices(neighbors, probabilities, k=1)[0]

# 更新信息素
def update_pheromones(paths):
    for edge in pheromones:
        pheromones[edge] *= (1 - pheromone_evaporation)

    for path in paths:
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            reverse_edge = (path[i + 1], path[i])
            if edge in pheromones:
                pheromones[edge] += pheromone_deposit
            elif reverse_edge in pheromones:
                pheromones[reverse_edge] += pheromone_deposit

# 应用蚁群算法
full_path = ant_colony_optimization(undirected_graph)

# 定义更新视图的函数
start_time = time.time()
animation_delay = 5  # 延迟时间（秒）

@viewer.on(interval=10)
def update_view(frame):
    if time.time() - start_time < animation_delay:
        return  # 如果未达到延迟时间，不执行任何操作

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