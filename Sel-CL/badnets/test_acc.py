import argparse
import os
import pathlib
import re
import sys
import time
import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import *
from deeplearning import evaluate_badnets, optimizer_picker, train_one_epoch
# from models import BadNet


from models.preact_resnet import *

parser = argparse.ArgumentParser(
    description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='CIFAR10', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--nb_classes', default=100, type=int, help='number of the classification types')
parser.add_argument('--load_local', action='store_true',
                    help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--epochs', default=100, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=0, help='Batch size to split dataset, default: 64')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate of the model, default: 0.001')
parser.add_argument('--download', action='store_true',
                    help='Do you want to download data ( default false, if you add this param, then download)')
parser.add_argument('--data_path', default='./data/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--device', default='cpu',
                    help='device to use for training / testing (cpu, or cuda:1, default: cpu)')
# poison settings
parser.add_argument('--poisoning_rate', type=float, default=1,
                    help='poisoning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--trigger_label', type=int, default=0,
                    help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
parser.add_argument('--trigger_path', default="../badnets/triggers/trigger_10.png",
                    help='Trigger Path (default: ./triggers/trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')
parser.add_argument('--train_root', default='./data', help='root for train data')
parser.add_argument('--num_classes', type=int, default=100, help='Number of in-distribution classes')
parser.add_argument('--noise_ratio', type=float, default=0.4, help='percent of noise')
parser.add_argument('--low_dim', type=int, default=128, help='Size of contrastive learning embedding')
parser.add_argument('--headType', type=str, default="Linear", help='Linear, NonLinear')
args = parser.parse_args()


def main():
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
    dataset_train, args.nb_classes = build_poisoned_training_set_original(is_train=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_testset_original(is_train=False, args=args)

    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)
    data_loader_val_clean = DataLoader(dataset_val_clean, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers)  # shuffle 随机化

    model = PreActResNet18(num_classes=args.num_classes, low_dim=args.low_dim, head=args.headType).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optimizer_picker(args.optimizer, model.parameters(), lr=args.lr)

    # basic_model_path = "./checkpoints/Sel-CL_model_CIFAR-10_0.0.pth"
    basic_model_path = r"/home/shixiong/Code/Sel-CL-main/badnets/checkpoints/Sel-CL_model_CIFAR-100_0.2.pth"
    start_time = time.time()
    if args.load_local:
        print("## Load model from : %s" % basic_model_path)

        model.load_state_dict(torch.load(basic_model_path).get("model"), strict=True)

        # model.training=False
        test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device, args.trigger_label)
        print(f"Test Clean Accuracy(TCA): {test_stats['clean_acc']:.4f}")
        print(f"Attack Success Rate(ASR): {test_stats['asr']:.4f}")
        print(f"Attack Success Rate New(ASR): {test_stats['asr_new']:.4f}")
    else:
        print(f"Start training for {args.epochs} epochs")
        stats = []
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(data_loader_train, model, criterion, optimizer, args.loss, device, args.trigger_label)
            test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device, args.trigger_label)
            print(
                f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f} Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}\n")

            # save model
            torch.save(model.state_dict(), basic_model_path)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         }

            # save training stats
            stats.append(log_stats)
            df = pd.DataFrame(stats)
            df.to_csv("./logs/%s_trigger%d.csv" % (args.dataset, args.trigger_label), index=False, encoding='utf-8')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()
