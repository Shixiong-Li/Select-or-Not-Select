import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import time
from dataset.cifar_dataset import *

import torch.utils.data as data
from torch import optim
from torchvision import datasets, transforms, models
import random
import sys

# sys.path.append('../utils')
sys.path.append(r'/home/shixiong/Code/Sel-CL-main/utils')
from utils_noise import *
from test_eval import test_eval
from queue_with_pro import *
from kNN_test import kNN
from MemoryMoCo import MemoryMoCo
from other_utils import *
from models.preact_resnet import *
from lr_scheduler import get_scheduler
from apex import amp
import argparse
from pprint import pformat
import numpy as np
import torch
import time
import logging
import sys, yaml, os
sys.path.append('../')
import re
import pathlib
from badnets.dataset import build_poisoned_training_set, build_testset, build_testset_original
from torch.utils.data import DataLoader
def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--epoch', type=int, default=200, help='training epoches')
    parser.add_argument('--warmup_way', type=str, default="uns", help='uns, sup')
    parser.add_argument('--warmup-epoch', type=int, default=1, help='warmup epoch')
    parser.add_argument('--lr', '--base-learning-rate', '--base-lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--lr-warmup-epoch', type=int, default=1, help='warmup epoch')
    parser.add_argument('--lr-warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr-decay-epochs', type=int, default=[125, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--initial_epoch', type=int, default=1, help="Star training at initial_epoch")

    parser.add_argument('--batch_size', type=int, default=128, help='#images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=100, help='#images in each mini-batch')
    parser.add_argument('--cuda_dev', type=int, default=1, help='GPU to select')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of in-distribution classes')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--dataset', type=str, default='CIFAR-10', help='CIFAR-10, CIFAR-100')
    parser.add_argument('--noise_type', default='asymmetric', help='symmetric or asymmetric')
    parser.add_argument('--noise_ratio', type=float, default=0.4, help='percent of noise')
    parser.add_argument('--train_root', default='./dataset', help='root for train data')
    parser.add_argument('--out', type=str, default='./out', help='Directory of the output')
    parser.add_argument('--experiment_name', type=str, default='Proof',
                        help='name of the experiment (for the output files)')
    parser.add_argument('--download', type=bool, default=False, help='download dataset')

    parser.add_argument('--network', type=str, default='PR18', help='Network architecture')
    parser.add_argument('--headType', type=str, default="Linear", help='Linear, NonLinear')
    parser.add_argument('--low_dim', type=int, default=128, help='Size of contrastive learning embedding')
    parser.add_argument('--seed_initialization', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--seed_dataset', type=int, default=42, help='random seed (default: 1)')
    parser.add_argument('--DA', type=str, default="complex", help='Choose simple or complex data augmentation')

    parser.add_argument('--alpha_m', type=float, default=1.0, help='Beta distribution parameter for mixup')
    parser.add_argument('--alpha_moving', type=float, default=0.999, help='exponential moving average weight')
    parser.add_argument('--alpha', type=float, default=0.5, help='example selection th')
    parser.add_argument('--beta', type=float, default=0.5, help='pair selection th')
    parser.add_argument('--uns_queue_k', type=int, default=10000, help='uns-cl num negative sampler')
    parser.add_argument('--uns_t', type=float, default=0.1, help='uns-cl temperature')
    parser.add_argument('--sup_t', default=0.1, type=float, help='sup-cl temperature')
    parser.add_argument('--sup_queue_use', type=int, default=1, help='1: Use queue for sup-cl')
    parser.add_argument('--sup_queue_begin', type=int, default=3, help='Epoch to begin using queue for sup-cl')
    parser.add_argument('--queue_per_class', type=int, default=100,
                        help='Num of samples per class to store in the queue. queue size = queue_per_class*num_classes*2')
    parser.add_argument('--aprox', type=int, default=1,
                        help='Approximation for numerical stability taken from supervised contrastive learning')
    parser.add_argument('--lambda_s', type=float, default=0.01, help='weight for similarity loss')
    parser.add_argument('--lambda_c', type=float, default=1, help='weight for classification loss')
    parser.add_argument('--k_val', type=int, default=250, help='k for k-nn correction')
    #############
    # parser.add_argument('--dataset', default='CIFAR10', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
    parser.add_argument('--nb_classes', default=100, type=int, help='number of the classification types')
    parser.add_argument('--load_local', action='store_true',
                        help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
    parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
    parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
    parser.add_argument('--epochs', default=100, help='Number of epochs to train backdoor model, default: 100')
    # parser.add_argument('--batch_size', type=int, default=64, help='Batch size to split dataset, default: 64')
    parser.add_argument('--num_workers', type=int, default=8, help='Batch size to split dataset, default: 64')
    # parser.add_argument('--lr', type=float, default=0.01, help='Learning rate of the model, default: 0.001')
    # parser.add_argument('--download', action='store_true',
    #                     help='Do you want to download data ( default false, if you add this param, then download)')
    parser.add_argument('--data_path', default='./data/', help='Place to load dataset (default: ./dataset/)')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing (cpu, or cuda:0, default: cpu)')
    # poison settings
    parser.add_argument('--poisoning_rate', type=float, default=0.05,
                        help='poisoning portion (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--trigger_label', type=int, default=0,
                        help='The NO. of trigger label (int, range from 0 to 9, default: 0)')
    parser.add_argument('--trigger_path', default="/home/shixiong/Code/Sel-CL-main/badnets/triggers/trigger_10.png",
                        help='Trigger Path (default: ./triggers/trigger_white.png)')
    parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')
    args = parser.parse_args()
    return args




def Generate(transform_train, transform_test):
    print("{}".format(args).replace(', ', ',\n'))

    if re.match('cuda:\d', args.device):
        cuda_num = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_num
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")  # if you're using MBP M1, you can also use "mps"

    # create related path
    pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True)

    print("\n# load dataset: %s " % args.dataset)
    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, args=args, transform_train=transform_train, transform_test=transform_test)
    dataset_val_clean, dataset_val_poisoned = build_testset_original(is_train=False, args=args)

    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True)
    data_loader_val_clean = DataLoader(dataset_val_clean, batch_size=args.test_batch_size, shuffle=False,
                                       num_workers=args.num_workers, pin_memory=True)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers)  # shuffle 随机化
    data_loader_train.dataset.targets = np.array(data_loader_train.dataset.targets)
    data_loader_val_clean.dataset.targets = np.array(data_loader_val_clean.dataset.targets)
    return data_loader_train, data_loader_val_clean, dataset_train

def data_config(args, transform_train, transform_test):
    # aa=np.load("../label_consistent_attacks/data/adv_dataset/cifar_resnet_e8_a1.5_s100.npz")["data"]
    # trainset_origin, testset_origin= get_dataset(args, TwoCropTransform(transform_train), transform_test)
    #
    #
    # train_loader_origin = torch.utils.data.DataLoader(trainset_origin, batch_size=args.batch_size, shuffle=True, num_workers=8,
    #                                            pin_memory=True)
    # test_loader_origin = torch.utils.data.DataLoader(testset_origin, batch_size=args.test_batch_size, shuffle=False, num_workers=8,
    #                                           pin_memory=True)
    train_loader, test_loader, trainset = Generate(TwoCropTransform(transform_train), transform_test)
    print('############# Data loaded #############')
    # for batch_idx  in enumerate(train_loader_origin):
    #     print(batch_idx)
    # for batch_idx in enumerate(train_loader):
    #     print(batch_idx)
    # return train_loader_origin, test_loader_origin, trainset_origin
    return train_loader, test_loader, trainset


def build_model(args, device):
    model = PreActResNet18(num_classes=args.num_classes, low_dim=args.low_dim, head=args.headType).to(device)
    model_ema = PreActResNet18(num_classes=args.num_classes, low_dim=args.low_dim, head=args.headType).to(device)

    # copy weights from `model' to `model_ema'
    moment_update(model, model_ema, 0)
    return model, model_ema


def main(args):
    exp_path = os.path.join(args.out, 'noise_models_' + args.network + '_{0}_SI{1}_SD{2}'.format(args.experiment_name,
                                                                                                 args.seed_initialization,
                                                                                                 args.seed_dataset),
                            args.noise_type, str(args.noise_ratio))
    res_path = os.path.join(args.out, 'metrics' + args.network + '_{0}_SI{1}_SD{2}'.format(args.experiment_name,
                                                                                           args.seed_initialization,
                                                                                           args.seed_dataset),
                            args.noise_type, str(args.noise_ratio))

    if not os.path.isdir(res_path):
        os.makedirs(res_path)

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)

    __console__ = sys.stdout
    name = "/results"
    log_file = open(res_path + name + ".log", 'a')
    sys.stdout = log_file
    print(args)

    args.best_acc = 0
    best_acc5 = 0
    best_acc_val = 0.0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_dev)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed_initialization)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed_initialization)  # GPU seed

    random.seed(args.seed_initialization)  # python seed for image transformation

    if args.dataset == 'CIFAR-10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'CIFAR-100' or args.dataset == 'CIFAR100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif args.dataset == 'MNIST':
        mean = [0.1307]
        std = [0.3081]

    if args.DA == "complex":
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # data loader
    num_classes = args.num_classes

    train_loader, test_loader, trainset = data_config(args, transform_train, transform_test)

    model, model_ema = build_model(args, device)
    uns_contrast = MemoryMoCo(args.low_dim, args.uns_queue_k, args.uns_t, thresh=0).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", num_losses=2)
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    if args.sup_queue_use == 1:
        queue = queue_with_pro(args, device)
    else:
        queue = []

    np.save(res_path + '/' + str(args.noise_ratio) + '_noisy_labels.npy', np.asarray(trainset.noisy_labels))

    for epoch in range(args.initial_epoch, args.epoch + 1):
        st = time.time()
        print("=================>    ", args.experiment_name, args.noise_ratio)
        if (epoch <= args.warmup_epoch):
            if (args.warmup_way == 'uns'):
                train_uns(args, scheduler, model, model_ema, uns_contrast, queue, device, train_loader, optimizer,
                          epoch)
            else:
                train_selected_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=4,
                                                                    pin_memory=True,
                                                                    sampler=torch.utils.data.WeightedRandomSampler(
                                                                        torch.ones(len(trainset)), len(trainset)))
                trainNoisyLabels = torch.LongTensor(train_loader.dataset.targets).unsqueeze(1)
                train_sup(args, scheduler, model, model_ema, uns_contrast, queue, device, train_loader,
                          train_selected_loader, optimizer, epoch, torch.eq(trainNoisyLabels, trainNoisyLabels.t()))
        else:
            train_selected_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=4,
                                                                pin_memory=True,
                                                                sampler=torch.utils.data.WeightedRandomSampler(
                                                                    selected_examples, len(selected_examples)))
            train_sel(args, scheduler, model, model_ema, uns_contrast, queue, device, train_loader,
                      train_selected_loader, optimizer, epoch, selected_pairs)

        if (epoch >= args.warmup_epoch):
            print('######## Pair-wise selection ########')
            selected_examples, selected_pairs,index_selected = pair_selection(args, model, device, train_loader, test_loader, epoch)

        print('Epoch time: {:.2f} seconds\n'.format(time.time() - st))
        index_poison = train_loader.dataset.poi_indices
        index_selected = index_selected.numpy().tolist()
        list_poison_selected = list(set(index_poison) & set(index_selected))
        print('Number of poisioned data choosen is {0}'.format(len(list_poison_selected)))
        print('Poisioned data choosen rate in selected example is {:.4f}'.format(len(list_poison_selected)/len(index_selected)))
        test_eval(args, model, device, test_loader)

        acc, acc5 = kNN(args, epoch, model, None, train_loader, test_loader, 200, 0.1, True)
        if acc >= args.best_acc:
            args.best_acc = acc
            best_acc5 = acc5
        print('KNN top-1 precion: {:.4f} {:.4f}, best is: {:.4f} {:.4f}'.format(acc * 100., \
                                                                                acc5 * 100., args.best_acc * 100.,
                                                                                best_acc5 * 100))

        if (epoch % 10 == 0):
            save_model(model, optimizer, args, epoch, exp_path + "/Sel-CL_model_"+args.dataset +"_"+ str(args.poisoning_rate) +".pth")
            np.save(res_path + '/' + 'selected_examples_train.npy', selected_examples.data.cpu().numpy())

        elif (epoch == 5):
            save_model(model, optimizer, args, epoch,
                       exp_path + "/surrogate_model_" + args.dataset + "_" + str(args.poisoning_rate) + ".pth")
            np.save(res_path + '/' + 'selected_examples_train.npy', selected_examples.data.cpu().numpy())
        log_file.flush()


if __name__ == "__main__":

    print(torch.cuda.current_device())
    args = parse_args()
    main(args)
