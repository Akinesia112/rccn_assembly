import random

# 你的 JSON 数据，这里简化为邻接表形式
graph = {
    "adjacency": {
        "0": {"1": None, "2": None},
        "1": {"0": None, "3": None},
        "2": {"0": None, "3": None},
        "3": {"1": None, "2": None}
        # 其他节点和边的定义
    },
    # 其他可能的数据结构
}

# 定义一个基因表示，这里使用砖块的顺序作为基因
def create_individual():
    return list(graph["adjacency"].keys())

# 定义适应度函数，根据具体问题来评估解的质量
def evaluate_fitness(individual):
    # 在这里，可以根据砖块的位置关系和接触面力学向量等信息计算适应度
    # 这里简化为路径长度的和，越小越好
    return sum([calculate_distance(individual[i], individual[i+1]) for i in range(len(individual)-1)])

def calculate_distance(node1, node2):
    # 模拟计算两个节点之间的距离
    # 在实际应用中，可以使用砖块的位置关系和接触面力学向量等信息
    return abs(int(node1) - int(node2))

# 定义交叉操作，这里使用交叉点之前的基因来生成新的个体
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
    return child

# 定义变异操作，这里使用随机交换两个基因
def mutate(individual):
    mutation_point1, mutation_point2 = random.sample(range(len(individual)), 2)
    individual[mutation_point1], individual[mutation_point2] = individual[mutation_point2], individual[mutation_point1]
    return individual

# 定义基因算法主体
def genetic_algorithm(population_size, generations):
    population = [create_individual() for _ in range(population_size)]

    for generation in range(generations):
        # 评估每个个体的适应度
        fitness_scores = [evaluate_fitness(individual) for individual in population]

        # 选择优秀个体
        selected_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k])[:population_size // 2]
        selected_population = [population[i] for i in selected_indices]

        # 生成新一代个体
        new_population = selected_population.copy()
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(selected_population, k=2)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
            new_population.append(child)

        population = new_population

    # 返回最优解
    best_solution = min(population, key=lambda x: evaluate_fitness(x))
    return best_solution

# 示例使用基因算法找到最优解
population_size = 50
generations = 100
mutation_rate = 0.1

best_solution = genetic_algorithm(population_size, generations)

print("Best Solution:", best_solution)
