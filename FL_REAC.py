import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import model_2


# 定义自动编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 定义训练函数
def train_autoencoder(params_dict, encoding_dim, num_epochs, device):
    # 创建数据集和数据加载器
    dataset = ModelWeightsDataset(params_dict)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # 确定输入维度
    input_dim = dataset.input_dim

    # 定义和训练自动编码器
    autoencoder = Autoencoder(input_dim, encoding_dim).to(device)
    Loss = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        for weights in data_loader:
            weights = weights.to(device)
            optimizer.zero_grad()
            inputs = weights.clone()
            outputs = autoencoder(inputs)
            loss = Loss(outputs, inputs)
            loss.backward()
            optimizer.step()

    return autoencoder


class ModelWeightsDataset(torch.utils.data.Dataset):
    def __init__(self, params_dict):
        self.weights = []
        for trainer_id, trainer_weights in params_dict.items():
            flattened_weights = torch.cat([v.flatten() for v in trainer_weights.values()])
            self.weights.append(flattened_weights)
        self.input_dim = self.weights[0].size(0)  # 保存输入维度

    def __len__(self):
        return len(self.weights)

    def __getitem__(self, idx):
        return self.weights[idx]



def cal_reconstruction_error(autoencoder, params_dict):
    reconstruction_errors = {}

    for trainer_id, trainer_weights in params_dict.items():
        weights_tensor = torch.cat([torch.flatten(v) for v in trainer_weights.values()])
        reconstructed_weights = autoencoder(weights_tensor)
        mse_loss = nn.MSELoss()
        mse = mse_loss(weights_tensor, reconstructed_weights)
        reconstruction_errors[trainer_id] = mse.item()

    return reconstruction_errors



def cal_mu_sigma(params_dict, device):
    # 计算mu_fi：权重误差前50%的均值
    encoding_dim = 50
    num_epochs = 10

    autoencoder = train_autoencoder(params_dict, encoding_dim, num_epochs, device)
    reconstruction_errors = cal_reconstruction_error(autoencoder, params_dict)
    errors_values = list(reconstruction_errors.values())
    sorted_errors_values = sorted(errors_values)  # 按照从小到大的顺序排序列表
    minimum_error = sorted_errors_values[0]  # 重构误差的最小值
    num_values_to_include = int(len(sorted_errors_values) * 0.5)  # 计算前50%最小值的个数
    mu_fi = mean_of_smallest_values = np.mean(sorted_errors_values[:num_values_to_include])  # 计算前50%最小值的均值
    sigma_fi = mu_fi - minimum_error

    return sigma_fi, mu_fi, minimum_error, reconstruction_errors


def anomaly_score(params_dict, device):
    A_i_dict = dict()
    beta = 2
    sigma_fi, mu_fi, minimum_error, reconstruction_errors = cal_mu_sigma(params_dict, device)

    a_i_dict = dict()
    for id in reconstruction_errors.keys():
        if reconstruction_errors[id] <= mu_fi + 3 * sigma_fi:
            a_ix = 1
        else:
            a_ix = reconstruction_errors[id] / minimum_error
        a_i_dict[id] = a_ix

    for id in reconstruction_errors.keys():

        A_i = math.exp(beta * (1-a_i_dict[id]))
        A_i_dict[id] = A_i

    return A_i_dict


def cal_diff(trainer_idx_verifier_loss, loss_dict, trainer_idx):

    V_ij = 0
    for id in range(len(trainer_idx_verifier_loss)):
        V_ij += abs(trainer_idx_verifier_loss[id] - loss_dict[trainer_idx])

    G_ij = 0
    for idx in loss_dict.keys():
        for id in range(len(trainer_idx_verifier_loss)):
            G_ij += abs(trainer_idx_verifier_loss[id] - loss_dict[idx])

    diff_i = V_ij * len(trainer_idx_verifier_loss) / G_ij

    return diff_i


def trust_score(diff_i_dict, sum_Gi):
    diff_im = sum(diff_i_dict.values()) / sum_Gi
    if diff_im <= 1:
        R_i = 1
    else:
        R_i = (1 / diff_im) ** diff_im

    return R_i


def cal_agg_weight(round, loss_dict, params_dict, R_i_dict, trainer_dict, data_num_dict, device):
    A_i_dict = anomaly_score(params_dict, device)
    score_i_dict = dict()
    if round <= 40:
        for id in loss_dict.keys():
            score_i = (1 - loss_dict[id] / sum(loss_dict.values())) * A_i_dict[id]
            score_i_dict[id] = score_i
    else:
        for id in loss_dict.keys():
            score_i = (1 - loss_dict[id] / sum(loss_dict.values())) * A_i_dict[id] * R_i_dict[id]
            score_i_dict[id] = score_i

    weighted_data_num_sum = 0
    for trainer_idx in trainer_dict.keys():
        weighted_data_num_sum += data_num_dict[trainer_idx] * score_i_dict[trainer_idx]

    agg_weights = []
    for trainer_idx in trainer_dict.keys():
        agg_weights.append(data_num_dict[trainer_idx] * score_i_dict[trainer_idx] / weighted_data_num_sum)


    return agg_weights









