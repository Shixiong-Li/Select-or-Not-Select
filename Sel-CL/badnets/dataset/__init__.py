# from utils.other_utils import TwoCropTransform
from .poisoned_dataset import CIFAR10Poison,MNISTPoison,CIFAR10Poison_original, CIFAR10PoisonLabelConsistent,CIFAR10PoisonLabelConsistentOnlyAdv,CIFAR10PoisonLabelConsistentAdvTrigger,CIFAR100Poison
from torchvision import datasets, transforms
import torch 
import os 
import numpy as np

def build_init_data(dataname, download, dataset_path):
    if dataname == 'MNIST':
        train_data = datasets.MNIST(root=dataset_path, train=True, download=download)
        test_data  = datasets.MNIST(root=dataset_path, train=False, download=download)
    elif dataname == 'CIFAR10' or dataname == 'CIFAR-10':
        train_data = datasets.CIFAR10(root=dataset_path, train=True,  download=download)
        test_data  = datasets.CIFAR10(root=dataset_path, train=False, download=download)
    elif dataname == 'CIFAR100' or dataname == 'CIFAR-100':
        train_data = datasets.CIFAR100(root=dataset_path, train=True, download=download)
        test_data = datasets.CIFAR100(root=dataset_path, train=False, download=download)

    return train_data, test_data

def build_poisoned_training_set(is_train, args, transform_train, transform_test):
    transform, detransform = build_transform(args.dataset)
    # print("Transform = ", transform)
    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR-10':
        # trainset = CIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform_train, target_transform=transform_test)
        trainset = CIFAR10PoisonLabelConsistent(args, args.data_path, train=is_train, download=True, transform=transform_train,
                                 target_transform=transform_test)

        nb_classes = 10
        trainset.random_in_noise()
    elif args.dataset == 'MNIST':
        trainset = MNISTPoison(args, args.data_path, train=is_train, download=True , transform=transform_train,
                                 target_transform=transform_test)
        nb_classes = 10
    elif args.dataset == 'CIFAR100' or args.dataset == 'CIFAR-100':
        trainset = CIFAR100Poison(args, args.data_path, train=is_train, download=True, transform=transform_train, target_transform=transform_test)
        nb_classes = 100
        trainset.random_in_noise()
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)
    print(trainset)

    return trainset, nb_classes




def build_testset(is_train, args, transform_train, transform_test):
    transform, detransform = build_transform(args.dataset)
    # print("Transform = ", transform)

    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR-10':
        testset_clean = datasets.CIFAR10(args.data_path, train=is_train, download=True, transform=transform)
        testset_poisoned = CIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'MNIST':
        testset_clean = datasets.MNIST(args.data_path, train=is_train, download=True, transform=transform)
        testset_poisoned = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'CIFAR100' or args.dataset == 'CIFAR-100':
        testset_clean = datasets.CIFAR100(args.data_path, train=is_train, download=True, transform=transform)
        testset_poisoned = CIFAR100Poison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 100
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)
    print(testset_clean, testset_poisoned)

    return testset_clean, testset_poisoned


def build_transform(dataset):
    if dataset == 'CIFAR10' or dataset == 'CIFAR-10':
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif dataset == "MNIST":
        mean, std = (0.5,), (0.5,)
    elif dataset == 'CIFAR100' or dataset == 'CIFAR-100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    else:
        raise NotImplementedError()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    detransform = transforms.Normalize((-mean / std).tolist(),
                                       (1.0 / std).tolist())  # you can use detransform to recover the image

    return transform, detransform


def build_poisoned_training_set_original(is_train, args):
    transform, detransform = build_transform(args.dataset)
    # print("Transform = ", transform)

    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR-10':
        trainset = CIFAR10Poison_original(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'MNIST':
        trainset = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'CIFAR100' or args.dataset == 'CIFAR-100':
        trainset = CIFAR100Poison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 100
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)
    print(trainset)

    return trainset, nb_classes

def build_testset_original(is_train, args):
    transform, detransform = build_transform(args.dataset)
    # print("Transform = ", transform)

    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR-10':
        testset_clean = datasets.CIFAR10(args.data_path, train=is_train, download=True, transform=transform)
        testset_poisoned = CIFAR10Poison_original(args, args.data_path, train=is_train, download=True, transform=transform)
        # testset_poisoned = CIFAR10PoisonLabelConsistentOnlyAdv(args, args.data_path, train=is_train, download=True,transform=transform)
        # testset_poisoned = CIFAR10PoisonLabelConsistentAdvTrigger(args, args.data_path, train=is_train, download=True,transform=transform)
        nb_classes = 10
    elif args.dataset == 'MNIST':
        testset_clean = datasets.MNIST(args.data_path, train=is_train, download=True, transform=transform)
        testset_poisoned = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'CIFAR100' or args.dataset == 'CIFAR-100':
        testset_clean = datasets.CIFAR100(args.data_path, train=is_train, download=True, transform=transform)
        testset_poisoned = CIFAR100Poison(args, args.data_path, train=is_train, download=True, transform=transform)
        # testset_poisoned = CIFAR10PoisonLabelConsistentOnlyAdv(args, args.data_path, train=is_train, download=True,transform=transform)
        # testset_poisoned = CIFAR10PoisonLabelConsistentAdvTrigger(args, args.data_path, train=is_train, download=True,transform=transform)
        nb_classes = 100
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)
    print(testset_clean, testset_poisoned)

    return testset_clean, testset_poisoned