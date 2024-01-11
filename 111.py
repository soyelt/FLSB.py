import numpy as np
import matplotlib.pyplot as plt

# 1. 设置随机种子
np.random.seed(42)


# 2. 随机抽样
N = 10000
x = np.random.uniform(-1, 1, N)
y = np.random.uniform(-1, 1, N)
inside_circle = x**2 + y**2 <= 1

# 3. 计算面积
estimated_area = 4 * np.sum(inside_circle) / N
print(f"Estimated area of the circle: {estimated_area}")

# 4. 可视化
plt.scatter(x[inside_circle], y[inside_circle], c="blue", s=1)
plt.scatter(x[~inside_circle], y[~inside_circle], c="red", s=1)
plt.axis('equal')
plt.title("Monte Carlo Estimation of Circle's Area")
plt.show()