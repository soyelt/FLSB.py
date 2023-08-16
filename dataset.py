import torch
from torchvision import datasets, transforms
import numpy as np


def get_dataset(args):

    if args.dataset == 'cifar10':
        data_path = './data'
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transform_test)
        # n_train = len(train_dataset)

    elif args.dataset == 'mnist':
        data_path = './data'
        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(32),
            transforms.ToTensor()

        ])
        transform_test = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform_test)
        # n_train = len(train_dataset)

    # dict_trainers = iid_sampling(n_train, num_trainers, args.seed)

    return train_dataset, test_dataset
