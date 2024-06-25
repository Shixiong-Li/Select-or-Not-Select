import random
import time

import torch
import torchvision as tv
from typing import Callable, Optional
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, MNIST
import os


# import label_consistent_attacks.poison_train
class TriggerHandler(object):

    def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))
        self.trigger_label = trigger_label
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        img.save('/home/shixiong/Code/Sel-CL-main/badnets/original.jpg')
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))  # 右下
        # img.paste(self.trigger_img, (0, 0)) #左上
        # img.paste(self.trigger_img, (self.img_width - self.trigger_size, 0)) #右上
        # img.paste(self.trigger_img, (0, self.img_height - self.trigger_size)) #左下
        img.save('/home/shixiong/Code/Sel-CL-main/badnets/badnet.jpg')
        return img

    # def apply_random_decay(self, img):
    #     img.save('/home/shixiong/Code/Sel-CL-main/badnets/before.jpg')
    #     img_np = img.numpy()  # 将Tensor转换为NumPy数组
    #     shape = img_np.shape
    #     decay_factors = np.random.uniform(0, 0.5, size=shape)  # 为每个像素点生成独立的随机衰减因子
    #     img_np *= decay_factors  # 将图像数组与衰减因子相乘
    #     img.data = torch.tensor(np.uint8(img_np * 255))  # 更新原始图像数据
    #     img.save('/home/shixiong/Code/Sel-CL-main/badnets/after.jpg')
    #     return img


# aa = np.load("/home/shixiong22/code/Sel-CL-main/label_consistent_attacks/data/adv_dataset/cifar_resnet_e8_a1.5_s100.npz")["data"]
def __apply_random_decay__(img):
    img.save('/home/shixiong/Code/Sel-CL-main/badnets/original.jpg')
    img_np = np.array(img)
    img_np = img_np.astype(np.float32)
    shape = img_np.shape
    # decay_factors = np.random.uniform(0, 40, size=shape[2])  # 为每个通道生成独立的随机衰减因子
    # decay_factors = [-38.298443705771156, -63.979900619448365, 49.41939058468119]
    decay_factors = [-20, -20, -20]
    for channel in range(shape[2]):  # 对每个通道应用不同的随机衰减
        img_np[:, :, channel] += decay_factors[channel]
        # img_np[:, :, channel] -= 5
        # img_np[:, :, channel] *= 1.3
    img = Image.fromarray(np.uint8(img_np))
    img.save('/home/shixiong/Code/Sel-CL-main/badnets/color.jpg')
    return img

class CIFAR10Poison(tv.datasets.CIFAR10):
    # class CIFAR10Poison(aa):
    def __init__(
            self,
            args,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = True,
            sample_indexes=None,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()

        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width,
                                              self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

        self.root = os.path.expanduser(args.train_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or validation set

        self.args = args
        if sample_indexes is not None:
            self.data = self.data[sample_indexes]
            self.targets = list(np.asarray(self.targets)[sample_indexes])

        self.num_classes = self.args.num_classes
        self.in_index = []
        self.out_index = []
        self.noisy_indexes = []
        self.clean_indexes = []
        self.clean_labels = []
        self.noisy_labels = []
        self.out_data = []
        self.out_labels = []
        self.soft_labels = []
        self.labelsNoisyOriginal = []
        self._num = []
        self._count = 1
        self.prediction = []
        self.confusion_matrix_in = np.array([])
        self.confusion_matrix_out = np.array([])
        self.labeled_idx = []
        self.unlabeled_idx = []
        self.poisoned_index = []
        self.random_indices=[]
        # self.is_poisoned = []
        #
        #
        # From in ou split function:
        self.soft_labels = np.zeros((len(self.targets), self.num_classes), dtype=np.float32)
        self._num = int(len(self.targets) * self.args.noise_ratio)

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
            # is_poisoned = 0
            # NOTE: According to the threat model, the trigger should be put on the image before transform.
            # (The attacker can only poison the dataset)
            # 从文件中读取需要偏移的索引
            # with open('random_indices_0.4.txt', 'r') as f:
            #     self.random_indices = [int(idx) for idx in f.readlines()]
            if index in self.poi_indices:
            # if index in self.random_indices:
                target = self.trigger_handler.trigger_label
                # img = self.trigger_handler.put_trigger(img)
                img = __apply_random_decay__(img)
                # is_poisoned = 1
            if self.transform is not None:
                img = self.transform(img)

            # if self.target_transform is not None:
            #     target = self.target_transform(target)

            return img, target, index
        else:
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
            # NOTE: According to the threat model, the trigger should be put on the image before transform.
            # (The attacker can only poison the dataset)
            # if index in self.poi_indices:
            if index in self.random_indices:
                target = self.trigger_handler.trigger_label
                # img = self.trigger_handler.put_trigger(img)
                img = __apply_random_decay__(img)
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target
        ################# Random in-distribution noise #########################

    def random_in_noise(self):
        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)
        idxes = np.random.permutation(len(self.targets))
        clean_labels = np.copy(self.targets)
        noisy_indexes = idxes[0:0]
        clean_indexes = idxes[0:self._num]
        for i in range(len(idxes)):
            if i < self._num:
                self.soft_labels[idxes[i]][
                    self.targets[idxes[i]]] = 0  ## Remove soft-label created during label mapping
                # targets[idxes[i]] -> another category
                label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                # while(label_sym==self.targets[idxes[i]]):#To exclude the original label
                # label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                self.targets[idxes[i]] = label_sym
            self.soft_labels[idxes[i]][self.targets[idxes[i]]] = 1

        self.targets = np.asarray(self.targets, dtype=np.long)
        self.noisy_labels = np.copy(self.targets)
        self.noisy_indexes = noisy_indexes
        self.clean_labels = clean_labels
        self.clean_indexes = clean_indexes
        self.confusion_matrix_in = (np.ones((self.args.num_classes, self.args.num_classes)) - np.identity(
            self.args.num_classes)) \
                                   * (self.args.noise_ratio / (self.num_classes - 1)) + \
                                   np.identity(self.args.num_classes) * (1 - self.args.noise_ratio)
        print('clean_num', sum(self.noisy_labels == self.clean_labels))

    ##########################################################################

    ################# Random in-distribution noise #########################

    def random_in_noise(self):
        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)
        idxes = np.random.permutation(len(self.targets))
        clean_labels = np.copy(self.targets)
        noisy_indexes = idxes[0:0]
        clean_indexes = idxes[0:self._num]
        for i in range(len(idxes)):
            if i < self._num:
                self.soft_labels[idxes[i]][
                    self.targets[idxes[i]]] = 0  ## Remove soft-label created during label mapping
                # targets[idxes[i]] -> another category
                label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                # while(label_sym==self.targets[idxes[i]]):#To exclude the original label
                # label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                self.targets[idxes[i]] = label_sym
            self.soft_labels[idxes[i]][self.targets[idxes[i]]] = 1

        self.targets = np.asarray(self.targets, dtype=np.int64)
        self.noisy_labels = np.copy(self.targets)
        self.noisy_indexes = noisy_indexes
        self.clean_labels = clean_labels
        self.clean_indexes = clean_indexes
        self.confusion_matrix_in = (np.ones((self.args.num_classes, self.args.num_classes)) - np.identity(
            self.args.num_classes)) \
                                   * (self.args.noise_ratio / (self.num_classes - 1)) + \
                                   np.identity(self.args.num_classes) * (1 - self.args.noise_ratio)
        print('clean_num', sum(self.noisy_labels == self.clean_labels))

    ##########################################################################


# def __apply_random_decay__(img):
#     img.save('/home/shixiong/Code/Sel-CL-main/badnets/before.jpg')
#     img_np = np.array(img)
#     img_np = img_np.astype(np.float32)
#     shape = img_np.shape
#     decay_factors = np.random.uniform(0, 0.1, size=shape)  # 为每个像素点生成独立的随机衰减因子
#     #0.05
#     # decay_factors = [0.04912414, 0.1268571,  0.12125213]
#     # 0.2
#     # decay_factors = [0.02851197, 0.04420828, 0.08152009]
#     for channel in range(shape[2]):  # 对每个通道应用不同的随机衰减
#         img_np[:, :, channel] *= 1 - decay_factors[channel]
#     img = Image.fromarray(np.uint8(img_np))
#     img.save('/home/shixiong/Code/Sel-CL-main/badnets/after.jpg')
#     return img




# # 将 PIL 图像转换为 YCrCb 并衰减
# def __apply_random_decay__(img):
#     img.save('/home/shixiong/Code/Sel-CL-main/badnets/before.jpg')
#     # 转换为YCrCb格式
#     img_ycrcb = img.convert("YCbCr")
#
#     # 转换为numpy数组，并确保类型是float32
#     ycrcb_np = np.array(img_ycrcb).astype(np.float32)
#
#     # 确保图像至少有3个通道
#     if ycrcb_np.shape[2] < 3:
#         raise ValueError("Expected an image with at least 3 channels.")
#
#     # 随机生成衰减因子，每个通道一个
#     decay_factors = np.random.uniform(0, 0.1, size=3)
#
#     # 在YCrCb通道上应用衰减
#     for i in range(ycrcb_np.shape[2]):
#         # 应用衰减因子
#         ycrcb_np[:, :, i] *= (1 - decay_factors[i])
#
#     # 转换回RGB格式并返回
#     img = Image.fromarray(np.uint8(ycrcb_np.clip(0, 255)), mode="YCbCr").convert("RGB")
#     img.save('/home/shixiong/Code/Sel-CL-main/badnets/after.jpg')
#     return img


class CIFAR10PoisonLabelConsistent(tv.datasets.CIFAR10):

    def __init__(
            self,
            args,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = True,
            sample_indexes=None,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform,
                         download=download)
        # # load adversarial dataset
        adv_dataset_path = "/home/shixiong/Code/Sel-CL-main/CIFAR/data/fully_poisoned_training_datasets/inf_8.npy"
        self.advData = np.load(adv_dataset_path)

        self.width, self.height, self.channels = self.__shape_info__()

        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width,
                                              self.height)
        # record all index of trigger_label
        target_needed_index = []
        for index, target_needed in enumerate(self.targets):
            if target_needed == args.trigger_label:
                target_needed_index.append(index)

        self.poisoning_rate = args.poisoning_rate if train else 1.0
        # indices = range(len(target_needed_index))
        # 使用当前时间戳作为种子
        current_time = int(time.time())
        random.seed(current_time)
        self.poi_indices = random.sample(target_needed_index, k=int(len(target_needed_index) * self.poisoning_rate))
        print(
            f"Poison {len(self.poi_indices)} over {len(target_needed_index)} samples ( poisoning rate {self.poisoning_rate})")

        self.root = os.path.expanduser(args.train_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or validation set

        self.args = args
        if sample_indexes is not None:
            self.data = self.data[sample_indexes]
            self.targets = list(np.asarray(self.targets)[sample_indexes])

        self.num_classes = self.args.num_classes
        self.in_index = []
        self.out_index = []
        self.noisy_indexes = []
        self.clean_indexes = []
        self.clean_labels = []
        self.noisy_labels = []
        self.out_data = []
        self.out_labels = []
        self.soft_labels = []
        self.labelsNoisyOriginal = []
        self._num = []
        self._count = 1
        self.prediction = []
        self.confusion_matrix_in = np.array([])
        self.confusion_matrix_out = np.array([])
        self.labeled_idx = []
        self.unlabeled_idx = []
        self.poisoned_index = []
        # self.is_poisoned = []
        #
        #
        # From in ou split function:
        self.soft_labels = np.zeros((len(self.targets), self.num_classes), dtype=np.float32)
        self._num = int(len(self.targets) * self.args.noise_ratio)

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
            # is_poisoned = 0
            # NOTE: According to the threat model, the trigger should be put on the image before transform.
            # (The attacker can only poison the dataset)
            # 从文件中读取需要偏移的索引
            # with open('random_indices.txt', 'r') as f:
            #     random_indices = [int(idx) for idx in f.readlines()]
            # if index in self.poi_indices:
            # if index in random_indices:
            if index == 80:
                # if index % 2.5 == 0:
                # if index == 35973 or index == 47328 or index == 26840:
                #     path = '/home/shixiong/Code/Sel-CL-main/badnets/' + str(index)
                #     img.save(path + 'before.jpg')
                #     img1 = self.advData[index]
                #     img1 = Image.fromarray(np.uint8(img1))
                #     img1.save(path + 'after.jpg')

                imgAdv = self.advData[index]
                img = Image.fromarray(np.uint8(imgAdv))
                img.save("inf_8.jpg")
                # target = self.trigger_handler.trigger_label



                # img = self.trigger_handler.put_trigger(img)
                # 衰减
                # img = __apply_random_decay__(img)
            if self.transform is not None:
                img = self.transform(img)

            # if self.target_transform is not None:
            #     target = self.target_transform(target)

            return img, target, index
        else:
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
            # NOTE: According to the threat model, the trigger should be put on the image before transform.
            # (The attacker can only poison the dataset)
            with open('random_indices.txt', 'r') as f:
                random_indices = [int(idx) for idx in f.readlines()]
            # if index in self.poi_indices:
            if index in random_indices:
                # if index % 2.5 == 0:
                # img.save('/home/shixiong22/code/Sel-CL-main/badnets/before.jpg')
                # img1 = self.advData[index]
                # img1 = Image.fromarray(np.uint8(img1))
                # img1.save('/home/shixiong22/code/Sel-CL-main/badnets/after.jpg')
                #
                # imgAdv = self.advData[index]
                # img = Image.fromarray(np.uint8(imgAdv))
                # target = self.trigger_handler.trigger_label
                img = __apply_random_decay__(img)
                # img = self.trigger_handler.put_trigger(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

        ################# Random in-distribution noise #########################

    def random_in_noise(self):
        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)
        idxes = np.random.permutation(len(self.targets))
        clean_labels = np.copy(self.targets)
        noisy_indexes = idxes[0:0]
        clean_indexes = idxes[0:self._num]
        for i in range(len(idxes)):
            if i < self._num:
                self.soft_labels[idxes[i]][
                    self.targets[idxes[i]]] = 0  ## Remove soft-label created during label mapping
                # targets[idxes[i]] -> another category
                label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                # while(label_sym==self.targets[idxes[i]]):#To exclude the original label
                # label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                self.targets[idxes[i]] = label_sym
            self.soft_labels[idxes[i]][self.targets[idxes[i]]] = 1

        self.targets = np.asarray(self.targets, dtype=np.long)
        self.noisy_labels = np.copy(self.targets)
        self.noisy_indexes = noisy_indexes
        self.clean_labels = clean_labels
        self.clean_indexes = clean_indexes
        self.confusion_matrix_in = (np.ones((self.args.num_classes, self.args.num_classes)) - np.identity(
            self.args.num_classes)) \
                                   * (self.args.noise_ratio / (self.num_classes - 1)) + \
                                   np.identity(self.args.num_classes) * (1 - self.args.noise_ratio)
        print('clean_num', sum(self.noisy_labels == self.clean_labels))

    ##########################################################################


class CIFAR10PoisonLabelConsistentOnlyAdv(tv.datasets.CIFAR10):

    def __init__(
            self,
            args,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform,
                         download=download)
        # # load adversarial dataset
        adv_dataset_path = "/home/shixiong/Downloads/home/shixiong/Downloads/fully_poisoned_training_datasets/inf_32.npy"
        self.advData = np.load(adv_dataset_path)

        self.width, self.height, self.channels = self.__shape_info__()

        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width,
                                              self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            imgAdv = self.advData[index]
            img = Image.fromarray(np.uint8(imgAdv))
            target = self.trigger_handler.trigger_label
            # img = self.trigger_handler.put_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10PoisonLabelConsistentAdvTrigger(tv.datasets.CIFAR10):

    def __init__(
            self,
            args,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform,
                         download=download)
        # # load adversarial dataset
        adv_dataset_path = "/home/shixiong22/code/label-consistent-backdoor-code-main/fully_poisoned_training_datasets/inf_8.npy"
        self.advData = np.load(adv_dataset_path)

        self.width, self.height, self.channels = self.__shape_info__()

        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width,
                                              self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            imgAdv = self.advData[index]
            img = Image.fromarray(np.uint8(imgAdv))
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    ##########################################################################


class CIFAR10Poison_original(CIFAR10):

    def __init__(
            self,
            args,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform,
                         download=download)

        self.width, self.height, self.channels = self.__shape_info__()

        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width,
                                              self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img = __apply_random_decay__(img)
            # img = self.trigger_handler.put_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MNISTPoison(MNIST):

    def __init__(
            self,
            args,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height = self.__shape_info__()
        self.channels = 1
        self.noisy_labels = []
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width,
                                              self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class MNISTPoisonOriginal(MNIST):

    def __init__(
            self,
            args,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height = self.__shape_info__()
        self.channels = 1

        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width,
                                              self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class CIFAR100Poison(tv.datasets.CIFAR100):  # 修改这里使用 CIFAR100 而不是 CIFAR10
    def __init__(
            self,
            args,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = True,
            sample_indexes=None,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()

        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width,
                                              self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

        self.root = os.path.expanduser(args.train_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or validation set

        self.args = args
        if sample_indexes is not None:
            self.data = self.data[sample_indexes]
            self.targets = list(np.asarray(self.targets)[sample_indexes])

        self.num_classes = 100  # 修改为 CIFAR-100 的类别数量
        self.in_index = []
        self.out_index = []
        self.noisy_indexes = []
        self.clean_indexes = []
        self.clean_labels = []
        self.noisy_labels = []
        self.out_data = []
        self.out_labels = []
        self.soft_labels = []
        self.labelsNoisyOriginal = []
        self._num = []
        self._count = 1
        self.prediction = []
        self.confusion_matrix_in = np.array([])
        self.confusion_matrix_out = np.array([])
        self.labeled_idx = []
        self.unlabeled_idx = []
        self.poisoned_index = []
        self.random_indices = []

        self.soft_labels = np.zeros((len(self.targets), self.num_classes), dtype=np.float32)
        self._num = int(len(self.targets) * self.args.noise_ratio)

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
            if index in self.poi_indices:
                target = self.trigger_handler.trigger_label
                # img = __apply_random_decay__(img)
                img = self.trigger_handler.put_trigger(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, target, index
        else:
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
            if index in self.poi_indices:
                target = self.trigger_handler.trigger_label
                img = self.trigger_handler.put_trigger(img)
                # img = __apply_random_decay__(img)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

    def random_in_noise(self):
        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)
        idxes = np.random.permutation(len(self.targets))
        clean_labels = np.copy(self.targets)
        noisy_indexes = idxes[0:0]
        clean_indexes = idxes[0:self._num]
        for i in range(len(idxes)):
            if i < self._num:
                self.soft_labels[idxes[i]][
                    self.targets[idxes[i]]] = 0  ## Remove soft-label created during label mapping
                # targets[idxes[i]] -> another category
                label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                # while(label_sym==self.targets[idxes[i]]):#To exclude the original label
                # label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                self.targets[idxes[i]] = label_sym
            self.soft_labels[idxes[i]][self.targets[idxes[i]]] = 1

        self.targets = np.asarray(self.targets, dtype=np.long)
        self.noisy_labels = np.copy(self.targets)
        self.noisy_indexes = noisy_indexes
        self.clean_labels = clean_labels
        self.clean_indexes = clean_indexes
        self.confusion_matrix_in = (np.ones((self.args.num_classes, self.args.num_classes)) - np.identity(
            self.args.num_classes)) \
                                   * (self.args.noise_ratio / (self.num_classes - 1)) + \
                                   np.identity(self.args.num_classes) * (1 - self.args.noise_ratio)
        print('clean_num', sum(self.noisy_labels == self.clean_labels))