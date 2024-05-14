import numpy as np
from scipy.stats import uniform, randint
from scipy.spatial.distance import pdist, squareform
from typing import List

# 定义数据结构类 PMedianData
class PMedianData:
    def __init__(self, distances, demands, capacities, p):
        self.distances = distances
        self.demands = demands
        self.capacities = capacities
        self.p = p

# 定义随机数据生成器类 PMedianGenerator
class PMedianGenerator:
    def __init__(
        self,
        x=uniform(loc=0.0, scale=100.0),
        y=uniform(loc=0.0, scale=100.0),
        n=4,  # 使用固定的点数
        p=randint(low=1, high=5),  # p 可以是随机的
        demands=uniform(loc=0, scale=10),
        capacities=uniform(loc=20, scale=100),
        distances_jitter=uniform(loc=1.0, scale=0.2),
        demands_jitter=uniform(loc=1.0, scale=0.2),
        capacities_jitter=uniform(loc=1.0, scale=0.2),
        fixed=True,
    ):
        self.x = x
        self.y = y
        self.n = n
        self.p = p
        self.demands = demands
        self.capacities = capacities
        self.distances_jitter = distances_jitter
        self.demands_jitter = demands_jitter
        self.capacities_jitter = capacities_jitter
        self.fixed = fixed
        self.ref_data = None

    def generate(self, n_samples: int) -> List[PMedianData]:
        if not self.fixed or self.ref_data is None:
            # 第一次运行或不使用固定数据时生成新数据
            p = self.p.rvs()
            loc = np.array([(self.x.rvs(), self.y.rvs()) for _ in range(self.n)])
            distances = squareform(pdist(loc))
            demands = self.demands.rvs(size=self.n)
            capacities = self.capacities.rvs(size=self.n)
            self.ref_data = PMedianData(distances=distances, demands=demands, capacities=capacities, p=p)

        # 根据fixed决定是否重新生成或使用扰动
        instances = []
        for _ in range(n_samples):
            if self.fixed:
                # 应用扰动
                distances = self.ref_data.distances * self.distances_jitter.rvs(size=(self.n, self.n))
                distances = np.tril(distances) + np.triu(distances.T, 1)  # 保证距离矩阵的对称性
                demands = self.ref_data.demands * self.demands_jitter.rvs(self.n)
                capacities = self.ref_data.capacities * self.capacities_jitter.rvs(self.n)
            else:
                # 生成新的数据
                p = self.p.rvs()
                loc = np.array([(self.x.rvs(), self.y.rvs()) for _ in range(self.n)])
                distances = squareform(pdist(loc))
                demands = self.demands.rvs(size=self.n)
                capacities = self.capacities.rvs(size=self.n)

            instances.append(PMedianData(distances=distances.round(2), demands=demands.round(2), capacities=capacities.round(2), p=p))

        return instances
def generate_pmedian_data(num_instances):
    generator = PMedianGenerator()
    return generator.generate(num_instances)

# 封装数据生成器为一个函数

def generate_random_problems(num_instances, num_point):
    generator = PMedianGenerator(
        n=num_point,  # 确保使用固定点数
        p=randint(low=1, high=num_point + 1),  # 确保 p 在合理范围内
        demands=uniform(loc=0, scale=10),
        capacities=uniform(loc=20, scale=100),
        distances_jitter=uniform(loc=1.0, scale=0.0),
        demands_jitter=uniform(loc=1.0, scale=0.0),
        capacities_jitter=uniform(loc=1.0, scale=0.0),
        fixed=False
    )
    return generator.generate(num_instances)