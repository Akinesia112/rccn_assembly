import networkx as nx
import random
import math
import time
import matplotlib.pyplot as plt
from compas_assembly.datastructures import Assembly
import pathlib
import csv
import sys
import psutil
from compas_view2.app import App

# 载入组装
CWD = pathlib.Path(__file__).parent.absolute()
FILE = CWD.parent.parent / "output" / "assembly_interface_from_rhino_test_2.json"
assembly = Assembly.from_json(FILE)

# 初始化视图器
viewer = App(width=2000, height=1000)

# 转换为无向图
undirected_graph = assembly.graph.to_networkx().to_undirected()

# 用于存储已添加的砖块对象
added_blocks = {}

performance_metrics = {
    "Space Required": 0,  # 在代码中计算
    "Min Time": sys.maxsize,
    "Max Time": 0,
    "Total Time": sys.maxsize - 0,  
    "Solution Time Size and Complexity": sys.maxsize-0,  # 在代码中计算
    "Solution Space Size and Complexity": 0,  # 在代码中计算
    "Precision Requirements": 0,  # 在代码中计算
    "Resource Availability": f"CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%"
}

# Simulation annealing
def simulated_annealing(graph, iterations):
    start_time = time.time()  # start time of algorithm 
    current_path = list(graph.nodes())
    current_cost = path_cost(graph, current_path)
    temperature = 0.5
    cooling_rate = 0.99
    min_temperature = 0.01
    costs = []

    iteration = 0
    while temperature > min_temperature and iteration < iterations:
        for _ in range(iterations):
            new_path = current_path[:]
            i, j = sorted(random.sample(range(len(new_path)), 2))
            new_path[i], new_path[j] = new_path[j], new_path[i]

            new_cost = path_cost(graph, new_path)
            forces = calculate_forces_on_interfaces(assembly)

            # Adjust the path costs 
            force_adjustment = forces["contact"] + forces["compression"] + forces["tension"] + forces["friction"] + forces["resultant"]
            new_cost_adjusted = new_cost * (1 + force_adjustment)

            if acceptance_probability(current_cost, new_cost_adjusted, temperature) > random.random():
                current_path, current_cost = new_path, new_cost_adjusted
            costs.append(current_cost)
        iteration += iterations
        temperature *= cooling_rate 
    
    end_time = time.time()  # 算法结束时间
    total_time = end_time - start_time  # 总运行时间   

    return current_path, costs, total_time  # 返回路径和每次温度下的代价列表與总运行时间

# 计算路径代价
def path_cost(graph, path):
    cost = 0
    for i in range(len(path) - 1):
        cost += nx.shortest_path_length(graph, path[i], path[i + 1], weight='weight')
    return cost

# 接受概率
def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost < old_cost:
        return 1.0
    else:
        return math.exp((old_cost - new_cost) / temperature)
    
# 缓存接口力的计算结果
forces_cache = {}

# 计算接口上的各种力
def calculate_forces_on_interfaces(assembly):
    global forces_cache
    total_forces = {"contact": 0, "compression": 0, "tension": 0, "friction": 0, "resultant": 0}
    for node, attr in assembly.graph.nodes(data=True):
        interface = attr.get('interface')
        if interface:
            if interface not in forces_cache:
                # 计算并缓存结果
                forces_cache[interface] = {
                    "contact": sum(interface.contactforces()),
                    "compression": sum(interface.compressionforces()),
                    "tension": sum(interface.tensionforces()),
                    "friction": sum(interface.frictionforces()),
                    "resultant": sum(interface.resultantforce())
                }
            # 从缓存中获取结果
            total_forces["contact"] += forces_cache[interface]["contact"]
            total_forces["compression"] += forces_cache[interface]["compression"]
            total_forces["tension"] += forces_cache[interface]["tension"]
            total_forces["friction"] += forces_cache[interface]["friction"]
            total_forces["resultant"] += forces_cache[interface]["resultant"]
    return total_forces

# 运行实验并收集数据
def run_experiments(graph, iterations_list):
    results = {}
    for iter_count in iterations_list:
        path, costs, total_time = simulated_annealing(graph, iter_count)
        results[iter_count] = (costs, total_time)
    return results

# 绘制收敛图
def plot_convergence(results, save_path):
    fig, ax = plt.subplots()
    # Assuming you have the exact color codes, replace these with the correct ones
    colors = ['#7da1cf', '#ee6faa', '#808080', '#9c6795']  # blue, pink, gray, purple
    
    # Enable the grid with a light grey color and set it to be behind the plot elements
    ax.grid(True, color='#D3D3D3', linestyle='-', linewidth=0.5, zorder=0)
    
    # Plot with specified colors, half the default linewidth and 70% opacity
    for (iter_count, costs), color in zip(results.items(), colors):
        ax.plot(costs, color=color, label=f"Iterations: {iter_count}", linewidth=0.8, alpha=0.7, zorder=3)
    
    # Set the opacity of the axis labels and title
    ax.set_xlabel("Iteration", fontsize=10, alpha=0.7)
    ax.set_ylabel("Cost", fontsize=10, alpha=0.7)
    ax.set_title("Convergence Over Iterations of Simulated Annealing Algorithm", fontsize=10, alpha=0.7)
    ax.legend()

    # Set the opacity of all tick labels to 70%
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_alpha(0.7)
    
    plt.savefig(save_path)
    plt.show()

# 主逻辑
iterations_list = [100, 500, 1000, 5000]
results = run_experiments(undirected_graph, iterations_list)
save_path = "D:\\RCCN_Assembly\\rccn_assembly\\scripts\\Algorithm\\pictures\\SA_convergence_plot.png"
plot_convergence({k: v[0] for k, v in results.items()}, save_path)


all_performance_metrics = []

# 应用不同iteration的模拟退火算法
for i in iterations_list:

    # 记录内存使用情况以计算所需空间
    initial_memory_usage = psutil.Process().memory_info().rss
    
    full_path, _, total_time  = simulated_annealing(undirected_graph, i)  # 这里选择一个迭代次数
    # 计算内存使用增加量
    final_memory_usage = psutil.Process().memory_info().rss
    performance_metrics["Space Required"] = final_memory_usage - initial_memory_usage

    # 计算解空间的大小和复杂性
    performance_metrics["Solution Space Size and Complexity"] = len(undirected_graph.nodes()) + len(undirected_graph.edges())
    # 计算解時间的大小和复杂性: simulated_annealing()函数中已计算total_time
    performance_metrics["Solution Time Size and Complexity"] = total_time
    # Precision Requirements
    optimal_cost = path_cost(undirected_graph, full_path)
    forces = calculate_forces_on_interfaces(assembly)
    total_force = sum(forces.values())
    final_cost = optimal_cost * (1 + total_force)
    performance_metrics["Precision Requirements"] = abs(final_cost - optimal_cost) / optimal_cost

    # 将当前iteration的性能指标加入列表
    all_performance_metrics.append({
        "Iterations": i,
        "Min Time": performance_metrics["Min Time"],
        "Max Time": performance_metrics["Max Time"],
        "Solution Time Size and Complexity": performance_metrics["Total Time"],
        "Space Required": performance_metrics["Space Required"],
        "Solution Space Size and Complexity": performance_metrics["Solution Space Size and Complexity"],
        "Precision Requirements": performance_metrics["Precision Requirements"],
        "Resource Availability": performance_metrics["Resource Availability"]
    })

# 打印性能指标
for key, value in performance_metrics.items():
    print(f"{key}: {value}")

# 将所有性能指标记录到CSV文件
csv_file = r"D:\RCCN_Assembly\rccn_assembly\scripts\Algorithm\csv\performance_metrics_SA.csv"
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=all_performance_metrics[0].keys())
    writer.writeheader()
    for metrics in all_performance_metrics:
        writer.writerow(metrics)

# 定义更新视图的函数
start_time = time.time()
animation_delay = 5  # 延迟时间（秒）

@viewer.on(interval=10)
def update_view(frame):
    if time.time() - start_time < animation_delay:
        return

    if full_path:
        current_node = full_path.pop()
        block = assembly.graph.node_attribute(current_node, 'block')

        if current_node not in added_blocks:
            obj = viewer.add(block, opacity=0.5)
            added_blocks[current_node] = obj
    else:
        viewer.stop()

# 设置视图器的相机位置和缩放
viewer.view.camera.scale = 100
viewer.view.camera.distance = 200

# 运行视图器
viewer.run()