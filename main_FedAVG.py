import random

import numpy as np
import torch
import torch.utils
import argparse
import copy
import model_2
import matplotlib.pyplot as plt
import dataset
import agent
from verifiers import Verifier
from trainers import Trainer
from DPnoise import add_noise, add_extra_noise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="dataset we use")
    parser.add_argument('--num_classes', type=int, default=10, help="the number of classes")
    parser.add_argument('--agent', type=int, default=1, help='the number of agent')
    parser.add_argument('--trainer_num', type=int, default=40, help='the number of trainers')
    parser.add_argument('--verifier_num', type=int, default=20, help='the number of verifiers')
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum")
    parser.add_argument('--dp', type=bool, default=True, help="whether add differential noise")
    parser.add_argument('--dp_epsilon', type=float, default=0.4, help='differential privacy budget')
    parser.add_argument('--dp_epsilon1', type=float, default=0.4, help='differential privacy budget')
    parser.add_argument('--dp_delta', type=float, default=1e-5, help='differential privacy relaxation term')
    parser.add_argument('--dp_clip', type=float, default=300.0, help='differential privacy clip')
    parser.add_argument('--bs', type=int, default=64, help="batch size")
    parser.add_argument('--round', type=int, default=100, help="communication round")
    parser.add_argument('--labda', type=float, default=1.0, help="lambda")
    return parser.parse_args()


def aggregate_model(w):
    # 创建一个和第一个模型权重相同形状的全零张量
    global_model = copy.deepcopy(w[0])
    for k in global_model.keys():
        global_model[k] = torch.zeros_like(w[0][k])

    # 对每个模型的权重进行加权相加
    for i in range(len(w)):
        for k in global_model.keys():
            global_model[k] += w[i][k]

    for k in global_model.keys():
        global_model[k] = global_model[k] / len(w)

    return global_model


acc_list = []

if __name__ == '__main__':
    # parse args
    args = args_parser()
    print(args)

    # Store the pre-trained model
    if args.dataset == 'mnist':
        agent_model = model_2.MNIST_CNN_Net()
    else:
        agent_model = model_2.CIFAR_CNN_Net()
    # torch.save(agent_model.state_dict(), './model_weight.pth')
    malicious_indices = random.sample(range(40), 18)

    dataset_train, dataset_test = dataset.get_dataset(args=args)
    # 创建并初始化训练者和验证者字典（实例化）
    trainer_dict = {idx: Trainer(args, agent_model, dataset_train, dataset_test, idx, malicious_indices) for idx in range(args.trainer_num)}
    verifier_dict = {idx: Verifier(args, dataset_test, agent_model, id=idx) for idx in range(args.verifier_num)}
    if args.dataset == 'mnist':
        net = model_2.MNIST_CNN_Net()
    else:
        net = model_2.CIFAR_CNN_Net()

    for round in range(args.round):
        print("---------------Round: {}------------------".format(round))
        params_dict = dict()
        acc_dict = dict()
        loss_dict = dict()
        data_num_dict = dict()
        trainer_indices = random.sample(range(40), 18)

        for trainer_idx in trainer_dict.keys():

            trainer = trainer_dict[trainer_idx]
            w, acc, loss, data_num = trainer.local_update(device, round)

            # 添加噪声
            if args.dp:
                net.load_state_dict(w)
                if trainer_idx in trainer_indices:
                    noised_net = add_extra_noise(device, args, net, dataset_train)
                else:
                    noised_net = add_noise(device, args, net, dataset_train)
                noised_w = noised_net.state_dict()
                params_dict[trainer_idx] = noised_w

            else:
                params_dict[trainer_idx] = w

            acc_dict[trainer_idx] = acc
            loss_dict[trainer_idx] = loss
            data_num_dict[trainer_idx] = len(data_num)

            print("----Round: {}----  trainer: {} ----Train Acc: {:.2f}----- Loss: {:.4f}------".format(round, trainer_idx, acc, loss))

        params = list(params_dict.values())

        global_model = aggregate_model(params)
        # agent_model.load_state_dict(global_model)
        # torch.save(agent_model.state_dict(), './model_weight2.pth')
        verifier = agent.Agent(args, dataset_test, global_model)
        total_acc = verifier.evaluate()
        acc_list.append(total_acc)

        print(f'----total acc:{total_acc:.2f}----')

        with open('pic/backdoor_attack_com_45.txt', 'a') as f:
            f.write(str(total_acc) + '\n')

        for trainer_idx in trainer_dict.keys():
            trainer = trainer_dict[trainer_idx]
            trainer.set_model_params(global_model)

    plt.figure()
    plt.plot(acc_list, label='FedAVG')
    plt.title('Global Model Accuracy')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(False)
    plt.show()
