import numpy as np
from matplotlib import pyplot as plt

with open('noise_effect_cifar10.txt', 'r') as f:
    lines = f.readlines()

# 将字符串数据转换为浮点数
data = [float(line.strip()) for line in lines]


no_noise = data[:100]  # 前100行数据无噪声
noise055 = data[100:250]  # 100-200 行数据 epsilon=0.55
noise045 = data[250:300]  # 200-300 行数据 epsilon=0.45
noise035 = data[300:400]  # 300-400 行数据 epsilon=0.35

# 创建一个新的图像
plt.figure()

plt.plot(no_noise, label='without noise', color='blue', linestyle='-')
plt.plot(noise055, label='ϵ = 0.55', color='red', linestyle='--')
plt.plot(noise045, label='\u03F5 = 0.45', color='green', linestyle='-.')
plt.plot(noise035, label='\u03F5 = 0.35', color='orange', linestyle=':')

# x_ticks = np.arange(0, len(no_noise), 10)  # 这会生成一个0到len(no_noise)之间，步长为10的等差数列
# y_ticks = np.linspace(0, 1, 11)  # 这会生成一个0到1之间，包含11个数字的等差数列

# plt.xtick
# 添加轴标签和标题
plt.xlabel('round')
plt.ylabel('accuracy')
plt.title('Noise Effect on Accuracy')
plt.legend(loc='lower right')
# 显示图像
plt.show()

