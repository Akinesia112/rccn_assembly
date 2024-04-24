import networkx as nx
from compas_assembly.datastructures import Assembly
from compas_view2.app import App
from compas_view2.objects import Object
import pathlib
import time
import psutil
import csv

# 载入组装
CWD = pathlib.Path(__file__).parent.absolute()
FILE = CWD.parent.parent / "output" / "assembly_interface_from_rhino_test_2.json"
assembly = Assembly.from_json(FILE)

# 初始化视图器
viewer = App(width=2000, height=1000)

# 定义性能指标
performance_metrics_bfs = {
    "Space Required": 0,
    "Total Time": 0,
    "Resource Availability": f"CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%"
}

# BFS
undirected_graph = assembly.graph.to_networkx().to_undirected()
node_count = 0

# 开始记录时间和内存使用
start_time = time.time()
initial_memory_usage = psutil.Process().memory_info().rss

# 准备开始 BFS 遍历
bfs_queue = []
visited = set()

# 找到第一个起点
for cc in nx.connected_components(undirected_graph):
    start_node = next(iter(cc))
    bfs_queue.append(start_node)
    visited.add(start_node)
    break

# 用于存储已添加的砖块对象
added_blocks = {}

# 定义更新视图的函数
start_time = time.time()
animation_delay = 5  # 延迟时间（秒）

# 定义更新视图的函数
@viewer.on(interval=10)  # 每n毫秒调用一次函数
def update_view(frame):
    if time.time() - start_time < animation_delay:
        return  # 如果未达到延迟时间，不执行任何操作
    if bfs_queue:  # 如果队列中还有节点
        current_node = bfs_queue.pop(0)  # 取出当前节点
        block = assembly.graph.node_attribute(current_node, 'block')

        # 显示当前节点对应的砖块
        if current_node not in added_blocks:
            obj = viewer.add(block, opacity=0.5)
            added_blocks[current_node] = obj

        # 将当前节点的邻居加入队列
        for neighbor in undirected_graph.neighbors(current_node):
            if neighbor not in visited:
                bfs_queue.append(neighbor)
                visited.add(neighbor)
    else:
        viewer.stop()  # 没有更多节点，结束动画


# 计算运行时间和空间使用
end_time = time.time()
final_memory_usage = psutil.Process().memory_info().rss
performance_metrics_bfs["Total Time"] = end_time - start_time # unit: seconds
performance_metrics_bfs["Space Required"] = final_memory_usage - initial_memory_usage # unit: bytes

# 打印性能指标
for key, value in performance_metrics_bfs.items():
    print(f"{key}: {value}")

# 将所有性能指标记录到CSV文件
csv_file = r"D:\RCCN_Assembly\rccn_assembly\scripts\Algorithm\csv\performance_metrics_BFS.csv"
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=performance_metrics_bfs.keys())
    writer.writeheader()
    writer.writerow(performance_metrics_bfs)

# 设置视图器的相机位置和缩放
viewer.view.camera.scale = 1000
viewer.view.camera.position = [3000, 3000, 3000]
viewer.view.camera.distance = 300

# 运行视图器
viewer.run()