import numpy as np
from matplotlib import pyplot as plt

with open('epsilon_effect.txt', 'r') as f:
    lines = f.readlines()

# 将字符串数据转换为浮点数
data = [float(line.strip()) for line in lines]

# clip=300, trainer_num=40, verifier=10
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

plt.plot(no_noise, label='without noise', color='blue', linestyle='-', linewidth=1.0)
plt.plot(noise1, label='ϵ = 0.5', color='red', linestyle='--', linewidth=1.0)
plt.plot(noise2, label='ϵ = 0.4', color='green', linestyle='-.', linewidth=1.0)
plt.plot(noise3, label='ϵ = 0.3', color='orange', linestyle=':', linewidth=1.0)
# plt.plot(noise4, label='no noise', color='black', linestyle='solid')
# plt.plot(noise5, label='60% malicious node', color='m', linestyle='dashed')
# plt.plot(noise6, label='90% malicious node', color='deeppink', linestyle='solid')
# plt.plot(noise7, label='80% malicious 10 times attack', color='deeppink', linestyle='solid')
# plt.plot(noise8, label='80% malicious 10 times attack', color='deeppink', linestyle='-')
# plt.plot(noise9, label='60% malicious node old', color='blue', linestyle='-')
plt.xticks(np.arange(0, 101, 10))  # 设置从0到100，间隔为10的刻度
plt.ylim(50, 100)


# 添加轴标签和标题
plt.xlabel('round')
plt.ylabel('accuracy')
plt.title('different privacy budget effect on accuracy')
plt.legend(loc='lower right')
# 显示图像
plt.show()

