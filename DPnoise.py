import copy
import numpy as np
import torch


def add_noise(device, args, old_net, train_data):
    net = copy.deepcopy(old_net)
    net.to(device)
    # Calculate sensitivity
    sensitivity = cal_sensitivity(args.lr, args.dp_clip, len(train_data) / args.trainer_num)

    with torch.no_grad():
        for k, v in net.named_parameters():

            noise = Gaussian_Simple(epsilon=args.dp_epsilon, delta=args.dp_delta, sensitivity=sensitivity, size=v.shape)
            noise = torch.from_numpy(noise).to(device)
            v.data.add_(noise)

    return net


def add_extra_noise(device, args, old_net, train_data):
    net = copy.deepcopy(old_net)
    net.to(device)
    # Calculate sensitivity
    sensitivity = cal_sensitivity(args.lr, args.dp_clip, len(train_data) / args.trainer_num)

    with torch.no_grad():
        for k, v in net.named_parameters():

            noise = Gaussian_Simple(epsilon=args.dp_epsilon1, delta=args.dp_delta, sensitivity=sensitivity, size=v.shape)
            noise = torch.from_numpy(noise).to(device)
            v.data.add_(noise)


    return net

def Gaussian_Simple(epsilon, delta, sensitivity, size):
    noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    return np.random.normal(0, noise_scale, size=size)


def cal_sensitivity(lr, clip, dataset_size):
    return 2 * lr * clip / dataset_size
