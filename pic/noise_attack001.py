import numpy as np
from matplotlib import pyplot as plt

with open('noise_attack001.txt', 'r') as f:
    lines = f.readlines()

# 将字符串数据转换为浮点数
data = [float(line.strip()) for line in lines]

# clip=800, trainer_num=10, verifier=10
no_noise = data[:100]  # 前100行数据无攻击
noise1 = data[100:200]  # 100-200 行数据 10%恶意
noise2 = data[200:300]  # 200-300 行数据 20%恶意
noise3 = data[300:400]  # 300-400 行数据 40%恶意
noise4 = data[400:500]  # 400-500 行数据 50%恶意
noise5 = data[500:600]  # 500-600 行数据 60%恶意
noise6 = data[600:700]
noise7 = data[700:800]
noise8 = data[800:900]
noise9 = data[900:1000]

# 创建一个新的图像
plt.figure()

plt.plot(no_noise, label='without attack', color='blue', linestyle='-')
plt.plot(noise1, label='10% malicious node', color='red', linestyle='--')
plt.plot(noise2, label='20% malicious node', color='green', linestyle='-.')
plt.plot(noise3, label='40% malicious node', color='orange', linestyle=':')
plt.plot(noise4, label='50% malicious node', color='black', linestyle='solid')
plt.plot(noise5, label='60% malicious node', color='m', linestyle='dashed')
# plt.plot(noise6, label='90% malicious node', color='deeppink', linestyle='solid')
# plt.plot(noise7, label='80% malicious 10 times attack', color='deeppink', linestyle='solid')
# plt.plot(noise8, label='80% malicious 10 times attack', color='deeppink', linestyle='-')
# plt.plot(noise9, label='60% malicious node old', color='blue', linestyle='-')


# 添加轴标签和标题
plt.xlabel('round')
plt.ylabel('accuracy')
plt.title('Noise Effect on Accuracy')
plt.legend(loc='lower right')
# 显示图像
plt.show()

