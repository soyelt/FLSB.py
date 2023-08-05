import numpy as np
from matplotlib import pyplot as plt

with open('lambda_effect.txt', 'r') as f:
    lines = f.readlines()

# 将字符串数据转换为浮点数
data = [float(line.strip()) for line in lines]

# clip=800, trainer_num=10, verifier=10, 无攻击
no_noise = data[:100]
noise1 = data[100:200]
noise2 = data[200:300]
noise3 = data[300:400]
noise4 = data[400:500]
noise5 = data[500:600]
noise6 = data[600:700]
noise7 = data[700:800]
noise8 = data[800:900]
noise9 = data[900:1000]

# 创建一个新的图像
plt.figure()

plt.plot(no_noise, label='λ = 0.5', color='blue', linestyle='-')
plt.plot(noise1, label='λ = 1', color='red', linestyle='--')
plt.plot(noise2, label='λ = 2', color='green', linestyle='-.')
plt.plot(noise3, label='λ = 4', color='orange', linestyle=':')
# plt.plot(noise4, label='ϵ = 0.40', color='black', linestyle='solid')
# plt.plot(noise5, label='60% malicious node', color='m', linestyle='dashed')
# plt.plot(noise6, label='90% malicious node', color='deeppink', linestyle='solid')
# plt.plot(noise7, label='80% malicious 10 times attack', color='deeppink', linestyle='solid')
# plt.plot(noise8, label='80% malicious 10 times attack', color='deeppink', linestyle='-')
# plt.plot(noise9, label='60% malicious node old', color='blue', linestyle='-')


# 添加轴标签和标题
plt.xlabel('round')
plt.ylabel('accuracy')
plt.title('different privacy budget effect on accuracy')
plt.legend(loc='lower right')
# 显示图像
plt.show()

