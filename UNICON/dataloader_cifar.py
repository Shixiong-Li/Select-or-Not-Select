from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import os
from autoaugment import CIFAR10Policy, ImageNetPolicy
from torchnet.meter import AUCMeter
import torch.nn.functional as F
from Asymmetric_Noise import *
from sklearn.metrics import confusion_matrix


## If you want to use the weights and biases
# import wandb
# wandb.init(project="noisy-label-project", entity="....")


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class TriggerHandler(object):
    # def __apply_random_decay__(self, img):
    #     # img.save('/home/shixiong/Code/Sel-CL-main/badnets/before.jpg')
    #     img_np = np.array(img)
    #     img_np = img_np.astype(np.float32)
    #     shape = img_np.shape
    #     decay_factors = np.random.uniform(0, 255, size=shape[2])  # 为每个通道生成独立的随机衰减因子
    #     for channel in range(shape[2]):  # 对每个通道应用不同的随机衰减
    #         img_np[:, :, channel] -= decay_factors[channel]
    #         # img_np[:, :, channel] -= 20
    #         # img_np[:, :, channel] *= 1.3
    #     img = Image.fromarray(np.uint8(img_np))
    #     # img.save('/home/shixiong/Code/Sel-CL-main/badnets/after.jpg')
    #     return img

    def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))
        self.trigger_label = trigger_label
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        img.save('./before.jpg')
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
        img.save('./after.jpg')
        return img

    def __apply_random_decay__(self, img):
        # img.save('./original.jpg')
        img_np = np.array(img)
        img_np = img_np.astype(np.float32)
        shape = img_np.shape
        # decay_factors = np.random.uniform(0, 40, size=shape[2])  # 为每个通道生成独立的随机衰减因子
        # decay_factors = [-38.298443705771156, -63.979900619448365, 49.41939058468119]
        decay_factors = [-40, -40, -40]
        for channel in range(shape[2]):  # 对每个通道应用不同的随机衰减
            img_np[:, :, channel] += decay_factors[channel]
            # img_np[:, :, channel] -= 5
            # img_np[:, :, channel] *= 1.3
        img = Image.fromarray(np.uint8(img_np))
        # img.save('./color.jpg')
        return img


class cifar_dataset(Dataset):
    def __init__(self, dataset, sample_ratio, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[],
                 probability=[], log='', poisoning_rate=0, trigger_label=0,
                 trigger_path="/home/shixiong/Code/Sel-CL-main/badnets/triggers/trigger_10.png", trigger_size=5):

        self.r = r  # noise ratio
        self.sample_ratio = sample_ratio
        self.transform = transform
        self.mode = mode
        root_dir_save = root_dir
        # self.poisoning_rate = poisoning_rate
        self.trigger_label = trigger_label
        self.trigger_path = trigger_path
        self.trigger_size = trigger_size

        # # load adversarial dataset
        adv_dataset_path = "/home/shixiong/Code/Sel-CL-main/CIFAR/data/fully_poisoned_training_datasets/inf_8.npy"
        self.advData = np.load(adv_dataset_path)

        if dataset == 'cifar10':
            root_dir = './data/cifar10/'
            num_class = 10
        else:
            root_dir = './data/cifar100/'
            num_class = 100

        ## For Asymmetric Noise (CIFAR10)
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}

        num_sample = 50000
        self.class_ind = {}

        if self.mode == 'test' or mode == 'test_poison':
            if dataset == 'cifar10':
                test_dic = unpickle('%s/test_batch' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset == 'cifar100':
                root_dir = './data/cifar100/'
                test_dic = unpickle('%s/test' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']

        else:
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset == 'cifar100':
                train_dic = unpickle('%s/train' % root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            if os.path.exists(noise_file):
                noise_label = np.load(noise_file)['label']
                noise_idx = np.load(noise_file)['index']
                idx = list(range(50000))
                clean_idx = [x for x in idx if x not in noise_idx]
                for kk in range(num_class):
                    self.class_ind[kk] = [i for i, x in enumerate(noise_label) if x == kk]

            else:  ## Inject Noise
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r * 50000)
                noise_idx = idx[:num_noise]

                if noise_mode == 'asym':
                    if dataset == 'cifar100':
                        noise_label, prob11 = noisify_cifar100_asymmetric(train_label, self.r)
                    else:
                        for i in range(50000):
                            if i in noise_idx:
                                noiselabel = self.transition[train_label[i]]
                                noise_label.append(noiselabel)
                            else:
                                noise_label.append(train_label[i])
                else:
                    for i in range(50000):
                        if i in noise_idx:
                            if noise_mode == 'sym':
                                if dataset == 'cifar10':
                                    noiselabel = random.randint(0, 9)
                                elif dataset == 'cifar100':
                                    noiselabel = random.randint(0, 99)
                                noise_label.append(noiselabel)

                            elif noise_mode == 'pair_flip':
                                noiselabel = self.pair_flipping[train_label[i]]
                                noise_label.append(noiselabel)

                        else:
                            noise_label.append(train_label[i])

                print("Save noisy labels to %s ..." % noise_file)
                np.savez(noise_file, label=noise_label, index=noise_idx)
                for kk in range(num_class):
                    self.class_ind[kk] = [i for i, x in enumerate(noise_label) if x == kk]

            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label

            else:
                save_file = 'Clean_index_' + str(dataset) + '_' + str(noise_mode) + '_' + str(self.r) + '.npz'
                save_file = os.path.join(root_dir_save, save_file)

                if self.mode == "labeled":
                    pred_idx = np.zeros(int(self.sample_ratio * num_sample))
                    class_len = int(self.sample_ratio * num_sample / num_class)
                    size_pred = 0

                    ## Ranking-based Selection and Introducing Class Balance
                    for i in range(num_class):
                        class_indices = self.class_ind[i]
                        prob1 = np.argsort(probability[class_indices].cpu().numpy())
                        size1 = len(class_indices)

                        try:
                            pred_idx[size_pred:size_pred + class_len] = np.array(class_indices)[
                                prob1[0:class_len].astype(int)].squeeze()
                            size_pred += class_len
                        except:
                            pred_idx[size_pred:size_pred + size1] = np.array(class_indices)
                            size_pred += size1

                    pred_idx = [int(x) for x in list(pred_idx)]
                    np.savez(save_file, index=pred_idx)

                    ## Weights for label refinement
                    probability[probability < 0.5] = 0
                    self.probability = [1 - probability[i] for i in pred_idx]

                elif self.mode == "unlabeled":
                    pred_idx = np.load(save_file)['index']
                    idx = list(range(num_sample))
                    pred_idx_noisy = [x for x in idx if x not in pred_idx]
                    pred_idx = pred_idx_noisy

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]

        # backdoor attack
        self.width, self.height, self.channels = self.__shape_info__()

        self.trigger_handler = TriggerHandler(self.trigger_path, self.trigger_size, self.trigger_label, self.width,
                                              self.height)
        indices = None
        if self.mode == 'test_poison':
            self.poisoning_rate = 1.0
            indices = range(len(self.test_label))
        elif self.mode == 'test':
            self.poisoning_rate = 0
            indices = range(len(self.test_label))
        else:
            self.poisoning_rate = poisoning_rate
            # badnet
            # indices = range(len(self.noise_label))

            # label-consistent
            # record all index of trigger_label
            indices = []
            for index, target_needed in enumerate(self.noise_label):
                if target_needed == trigger_label:
                    indices.append(index)

        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    def __shape_info__(self):
        if self.mode == 'test' or self.mode == 'test_poison':
            return self.test_data.shape[1:]
        else:
            return self.train_data.shape[1:]

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            image = Image.fromarray(img)
            if index in self.poi_indices:
                target = self.trigger_handler.trigger_label
                # img = self.trigger_handler.__apply_rdanom_decay__(img)
                image.save("advBefore.jpg")
                imgAdv = self.advData[index]
                image = Image.fromarray(np.uint8(imgAdv))
                image = self.trigger_handler.put_trigger(image)
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)

            return img1, img2, img3, img4, target, prob

        elif self.mode == 'unlabeled':
            img = self.train_data[index]
            image = Image.fromarray(img)
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)
            return img1, img2, img3, img4

        elif self.mode == 'all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            if index in self.poi_indices:
                target = self.trigger_handler.trigger_label
                # img = self.trigger_handler.__apply_random_decay__(img)
                img.save("advBefore.jpg")
                imgAdv = self.advData[index]
                img = Image.fromarray(np.uint8(imgAdv))
                img = self.trigger_handler.put_trigger(img)
            img = self.transform(img)

            return img, target, index

        elif self.mode == 'test' or self.mode == 'test_poison':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)

            if index in self.poi_indices:
                target = self.trigger_handler.trigger_label
                # img = self.trigger_handler.__apply_random_decay__(img)
                img.save("advBefore.jpg")
                imgAdv = self.advData[index]
                img = Image.fromarray(np.uint8(imgAdv))
                img = self.trigger_handler.put_trigger(img)
            img = self.transform(img)

            return img, target

    def __len__(self):
        if self.mode != 'test' and self.mode != 'test_poison':
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifar_dataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file='', poisoning_rate=0,
                 trigger_label=0, trigger_path="/home/shixiong/Code/Sel-CL-main/badnets/triggers/trigger_10.png",
                 trigger_size=5):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        self.poisoning_rate = poisoning_rate
        self.trigger_label = trigger_label
        self.trigger_path = trigger_path
        self.trigger_size = trigger_size

        if self.dataset == 'cifar10':
            transform_weak_10 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            transform_strong_10 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            self.transforms = {
                "warmup": transform_weak_10,
                "unlabeled": [
                    transform_weak_10,
                    transform_weak_10,
                    transform_strong_10,
                    transform_strong_10
                ],
                "labeled": [
                    transform_weak_10,
                    transform_weak_10,
                    transform_strong_10,
                    transform_strong_10
                ],
            }

            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        elif self.dataset == 'cifar100':
            transform_weak_100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )

            transform_strong_100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )

            self.transforms = {
                "warmup": transform_weak_100,
                "unlabeled": [
                    transform_weak_100,
                    transform_weak_100,
                    transform_strong_100,
                    transform_strong_100
                ],
                "labeled": [
                    transform_weak_100,
                    transform_weak_100,
                    transform_strong_100,
                    transform_strong_100
                ],
            }
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])

    def run(self, sample_ratio, mode, pred=[], prob=[]):
        if mode == 'warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, sample_ratio=sample_ratio, noise_mode=self.noise_mode,
                                        r=self.r, root_dir=self.root_dir, transform=self.transforms["warmup"],
                                        mode="all", noise_file=self.noise_file, poisoning_rate=self.poisoning_rate,
                                        trigger_label=self.trigger_label, trigger_path=self.trigger_path,
                                        trigger_size=self.trigger_size)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, sample_ratio=sample_ratio, noise_mode=self.noise_mode,
                                            r=self.r, root_dir=self.root_dir, transform=self.transforms["labeled"],
                                            mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob,
                                            log=self.log, poisoning_rate=self.poisoning_rate,
                                            trigger_label=self.trigger_label, trigger_path=self.trigger_path,
                                            trigger_size=self.trigger_size)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers, drop_last=True)

            unlabeled_dataset = cifar_dataset(dataset=self.dataset, sample_ratio=sample_ratio,
                                              noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir,
                                              transform=self.transforms["unlabeled"], mode="unlabeled",
                                              noise_file=self.noise_file, pred=pred, poisoning_rate=self.poisoning_rate,
                                              trigger_label=self.trigger_label, trigger_path=self.trigger_path,
                                              trigger_size=self.trigger_size)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=int(self.batch_size / (2 * sample_ratio)),
                shuffle=False,
                num_workers=self.num_workers, drop_last=True)

            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'test' or mode == 'test_poison':
            test_dataset = cifar_dataset(dataset=self.dataset, sample_ratio=sample_ratio, noise_mode=self.noise_mode,
                                         r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode=mode,
                                         poisoning_rate=self.poisoning_rate, trigger_label=self.trigger_label,
                                         trigger_path=self.trigger_path, trigger_size=self.trigger_size)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=100,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        # elif mode=='test_poison':
        #     test_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test_poison', poisoning_rate = self.poisoning_rate, trigger_label = self.trigger_label, trigger_path=self.trigger_path, trigger_size=self.trigger_size)
        #     test_loader = DataLoader(
        #         dataset=test_dataset,
        #         batch_size=100,
        #         shuffle=False,
        #         num_workers=self.num_workers)
        #     return test_loader

        elif mode == 'eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, sample_ratio=sample_ratio, noise_mode=self.noise_mode,
                                         r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all',
                                         noise_file=self.noise_file, poisoning_rate=self.poisoning_rate,
                                         trigger_label=self.trigger_label, trigger_path=self.trigger_path,
                                         trigger_size=self.trigger_size)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=100,
                shuffle=False,
                num_workers=self.num_workers, drop_last=True)
            return eval_loader
