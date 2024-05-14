from PMedianGenerator import generate_random_problems
from MIP_solver import solve_pmedian_instance
import numpy as np

def training_data(num_instances=100, num_point=4, num_facilities=2):
    instances = generate_random_problems(num_instances, num_point)
    features = []
    targets_classification = []
    targets_regression = []
    for data in instances:
        # 特征
        instance_features = np.concatenate([
            data.distances.flatten(),
            data.demands,
            data.capacities,
            np.array([data.p])
        ])
        features.append(instance_features)

        # 使用Pyomo求解器得到的最优决策变量和成本值
        facility_decision, optimal_cost = solve_pmedian_instance(data, verbose=False)  # 训练时设置 verbose 为 False
        if facility_decision is None or optimal_cost is None:
            continue  # 跳过无效的实例

        # 展开嵌套列表并处理 None 值
        facility_decision_flat = np.array([int(val) if val is not None else 0 for sublist in facility_decision for val in sublist])
        targets_classification.append(facility_decision_flat)
        targets_regression.append([optimal_cost])

    X = np.array(features, dtype=np.float32)
    Y_class = np.array(targets_classification, dtype=np.float32)
    Y_regress = np.array(targets_regression, dtype=np.float32)
    return X, Y_class, Y_regress

# 循环生成数据并保存
num_instances = 10000
num_test = 1000

for num_point in range(3, 11):
    num_facilities = num_point * num_point

    X_train, Y_train_class, Y_train_regress = training_data(num_instances, num_point, num_facilities)
    X_test, Y_test_class, Y_test_regress = training_data(num_test, num_point, num_facilities)

    # 保存训练数据到本地
    np.save(f'X_train_b{num_point}.npy', X_train)
    np.save(f'Y_train_class_b{num_point}.npy', Y_train_class)
    np.save(f'Y_train_regress_b{num_point}.npy', Y_train_regress)
    np.save(f'X_test_b{num_point}.npy', X_test)
    np.save(f'Y_test_class_b{num_point}.npy', Y_test_class)
    np.save(f'Y_test_regress_b{num_point}.npy', Y_test_regress)

    print(f"训练和测试数据已保存到本地文件，num_point = {num_point}。")
