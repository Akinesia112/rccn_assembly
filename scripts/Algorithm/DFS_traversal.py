import networkx as nx
import time
import sys
import psutil
from compas_assembly.datastructures import Assembly
from compas_view2.app import App
import pathlib
import csv

# 载入组装
CWD = pathlib.Path(__file__).parent.absolute()
FILE = CWD.parent.parent / "output" / "assembly_interface_from_rhino_bulges.json"
assembly = Assembly.from_json(FILE)

# 初始化视图器
viewer = App(width=2000, height=1000)

# 转换为无向图
undirected_graph = assembly.graph.to_networkx().to_undirected()

# 性能指标初始化
performance_metrics_dfs = {
    "Space Required": 0,
    "Total Time": 0,
    "Resource Availability": f"CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%"
}

# DFS算法
def depth_first_search(graph, start_node):
    start_time = time.time()  # 开始计时
    initial_memory_usage = psutil.Process().memory_info().rss

    stack = [start_node]
    visited = set([start_node])
    path = []

    while stack:
        current_node = stack.pop()
        path.append(current_node)
        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                stack.append(neighbor)
                visited.add(neighbor)

    end_time = time.time()  # 结束计时
    final_memory_usage = psutil.Process().memory_info().rss

    total_time = end_time - start_time
    space_required = final_memory_usage - initial_memory_usage

    performance_metrics_dfs["Total Time"] = total_time
    performance_metrics_dfs["Space Required"] = space_required

    return path

# 运行DFS算法并记录性能指标
start_node = next(iter(undirected_graph.nodes))
dfs_path = depth_first_search(undirected_graph, start_node)

# 打印性能指标
for key, value in performance_metrics_dfs.items():
    print(f"{key}: {value}")

# 用于存储已添加的砖块对象
added_blocks = {}

# 定义更新视图的函数
@viewer.on(interval=10)
def update_view(frame):
    if dfs_path:
        current_node = dfs_path.pop(0)
        block = assembly.graph.node_attribute(current_node, 'block')

        if current_node not in added_blocks:
            obj = viewer.add(block, opacity=0.5)
            added_blocks[current_node] = obj
    else:
        viewer.stop()

# csv file
csv_file = r"D:\RCCN_Assembly\rccn_assembly\scripts\Algorithm\csv\performance_metrics_DFS.csv"
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=performance_metrics_dfs.keys())
    writer.writeheader()
    writer.writerow(performance_metrics_dfs)
# 设置视图器的相机位置和缩放
viewer.view.camera.scale = 1000
viewer.view.camera.position = [2000, 2000, 2000]
viewer.view.camera.distance = 300

# 运行视图器
viewer.run()
