import math
import statistics

import numpy as np
import torch
import model_2



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Agent(object):
    def __init__(self, args, eva_dataset, global_model):
        if args.dataset == 'mnist':
            self.model = model_2.MNIST_CNN_Net()
        else:
            self.model = model_2.CIFAR_CNN_Net()
        self.eva_loader = torch.utils.data.DataLoader(eva_dataset, batch_size=args.bs, shuffle=True)
        # self.model.load_state_dict(torch.load(model_path))
        self.model.load_state_dict(global_model)
        self.model.to(device)
    def evaluate(self):
        self.model.to(device)
        self.model.eval()
        correct = 0
        data_size = 0
        for batch_id, batch in enumerate(self.eva_loader):
            data, target = batch
            data_size += data.size()[0]
            data = data.to(device)
            target = target.to(device)

            output = self.model(data)

            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        eval_acc = 100.0 * (float(correct) / float(data_size))

        return eval_acc


def cal_diff(trainer_idx_verifier_loss, loss_dict, trainer_idx):
    beta = 1
    diff_i = 0

    loss_im = median = statistics.median(trainer_idx_verifier_loss)
    for id, loss_ij in enumerate(trainer_idx_verifier_loss):
        diff_i += abs(loss_ij - loss_dict[trainer_idx]) / math.exp(beta * (loss_ij - loss_im))

    # verifier_loss = np.array(trainer_idx_verifier_loss)
    # verifier_trainer_idx = np.abs(verifier_loss - loss_dict[trainer_idx])
    # verifier_loss = np.abs(verifier_loss - loss_im)
    #
    # diff_trainer_idx = np.sum(verifier_trainer_idx/np.exp(verifier_loss))  # 0.24065

    return diff_i


def cal_agg_weight(args, diff_trainer_dict, loss_dict, params_dict, trainer_dict, data_num_dict):

    diff_sum = 0
    for trainer_idx in diff_trainer_dict.keys():
        diff_sum += diff_trainer_dict[trainer_idx]

    loss_sum = 0
    for trainer_idx in loss_dict.keys():
        loss_sum += loss_dict[trainer_idx]

    A_dict = dict()
    S_dict = dict()
    for trainer_idx in trainer_dict.keys():
        S_dict[trainer_idx] = diff_sum / len(trainer_dict) / diff_trainer_dict[trainer_idx]
        A_dict[trainer_idx] = loss_sum / len(trainer_dict) / loss_dict[trainer_idx]

    D_dict = dict()
    for trainer_idx in trainer_dict.keys():
        if S_dict[trainer_idx] <= 1:
            d_i = 0
        else:
            d_i = A_dict[trainer_idx] * S_dict[trainer_idx] ** args.labda
        D_dict[trainer_idx] = d_i
    # D_dict = dict()
    # for trainer_idx in trainer_dict.keys():
    #     d_i = A_dict[trainer_idx] * float(np.exp(args.labda * S_dict[trainer_idx]))
    #     D_dict[trainer_idx] = d_i

    weighted_data_num_sum = 0
    for trainer_idx in trainer_dict.keys():
        weighted_data_num_sum += data_num_dict[trainer_idx] * D_dict[trainer_idx]

    agg_weights = []
    for trainer_idx in trainer_dict.keys():
        agg_weights.append(data_num_dict[trainer_idx] * D_dict[trainer_idx] / weighted_data_num_sum)

    params = list(params_dict.values())

    return params, agg_weights