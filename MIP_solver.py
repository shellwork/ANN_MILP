from pyomo.environ import *
from pyomo.opt import SolverFactory

def solve_pmedian_instance(data, verbose=False):
    model = ConcreteModel()
    I = set(range(len(data.demands)))  # 客户和设施的索引
    model.x = Var(I, I, domain=Binary)  # x[i, j]
    model.y = Var(I, domain=Binary)  # y[j]

    # Objective: Minimize total distance cost
    model.obj = Objective(expr=sum(data.distances[i][j] * model.x[i, j] for i in I for j in I), sense=minimize)

    # Constraint: Each customer is assigned to exactly one facility
    model.one_assignment = ConstraintList()
    for i in I:
        model.one_assignment.add(expr=sum(model.x[i, j] for j in I) == 1)

    # Constraint: Exactly p facilities are open
    model.num_facilities = Constraint(expr=sum(model.y[j] for j in I) == data.p)

    # Constraint: Capacity constraint for each facility
    model.capacity = ConstraintList()
    for j in I:
        model.capacity.add(expr=sum(data.demands[i] * model.x[i, j] for i in I) <= data.capacities[j] * model.y[j])

    solver = SolverFactory('glpk')
    results = solver.solve(model, tee=verbose)  # 设置 tee 参数为 verbose

    # 输出求解状态和结果 (仅在 verbose 为 True 时输出)
    if verbose:
        print(f'Status: {results.solver.status}')
        print(f'Termination condition: {results.solver.termination_condition}')
        print(f'Optimal value of objective function: {model.obj()}')

    # 收集解决方案
    facility_decision = [[int(model.x[i, j].value) for j in I] for i in I]
    optimal_cost = model.obj()

    return facility_decision, optimal_cost


# 示例调用函数
if __name__ == "__main__":
    from PMedianGenerator import generate_random_problems

    num_point = 4
    num_instances = 5  # 指定生成和求解的实例数量
    data_instances = generate_random_problems(num_instances, num_point)

    for idx, data in enumerate(data_instances):
        print(f"\nSolving instance {idx + 1}...")
        print(f"Demands: {data.demands}")
        print(f"Distances: {data.distances}")
        print(f"Capacities: {data.capacities}")
        print(f"Number of facilities to open: {data.p}")
        facility_decision, optimal_cost = solve_pmedian_instance(data, verbose=True)  # 在测试时设置 verbose 为 True
        print(f"Optimal cost: {optimal_cost}")
        print("Facility decision matrix:")
        for row in facility_decision:
            print(row)
