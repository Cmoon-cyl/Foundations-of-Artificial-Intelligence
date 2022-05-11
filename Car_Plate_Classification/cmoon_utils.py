#!/usr/bin/env python
# coding: UTF-8 
# Created by Cmoon

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2
import os
import time
from matplotlib import pyplot as plt


def plot_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            img = img.permute(1, 2, 0)
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def accuracy(y_hat, y):
    """计算正确的个数"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


def evaluate_accuracy(net, data_iter):
    """计算网络在某个数据集上的准确率"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    with torch.no_grad():
        for X, y in data_iter:
            X, y = try_gpu(X, y)
            metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]


class Accumulator:
    """累加器"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TboardWriter:
    """Tensorboard writer"""

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def write_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def write_image(self, tag, img, step):
        self.writer.add_image(tag, img, step)

    def close(self):
        self.writer.close()


def get_file_list(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                file_list.append(os.path.join(root, file))
    return file_list


def get_sub_dir(path):
    sub_dir = []
    for dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, dir)):
            sub_dir.append(dir)
    return sub_dir


def create_dir(path, dir_list):
    for dir in dir_list:
        if not os.path.exists(os.path.join(path, dir)):
            os.mkdir(os.path.join(path, dir))


class MyData(Dataset):
    def __init__(self, root, index: int):
        self.root_dir = root
        self.index = int(index)
        self.label = idx2label(self.index)
        self.path = os.path.join(self.root_dir, self.label)
        self.img_list = os.listdir(self.path)
        self.transform = transforms.Compose([transforms.ToTensor(), ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        label = self.index
        img_name = self.img_list[index]
        img_path = os.path.join(self.path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, label


def split(dataset, ratio):
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, test_size])
    return train_data, val_data


def load_datasets(batch_size=64, train_ratio=0.7, is_shuffle=True):
    train_dir = r'dataset/train'
    test_dir = r'dataset/test'
    train_path = get_sub_dir(train_dir)
    test_path = get_sub_dir(test_dir)
    train_label = label2idx(train_path)
    test_label = label2idx(test_path)
    # 初始化训练集,验证集和测试集(训练集和验证集73开)
    dataset = MyData(train_dir, train_label[0])
    train_dataset, val_dataset = split(dataset, train_ratio)
    test_dataset = MyData(test_dir, test_label[0])
    for label in train_label[1:]:
        dataset = MyData(train_dir, label)
        new_train, new_val = split(dataset, train_ratio)
        train_dataset += new_train
        val_dataset += new_val
    for label in test_label[1:]:
        dataset = MyData(test_dir, label)
        test_dataset += dataset
    print('训练集:{},验证集:{},测试集:{}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))
    return (DataLoader(train_dataset, batch_size, shuffle=is_shuffle, num_workers=2),
            DataLoader(val_dataset, batch_size, shuffle=is_shuffle, num_workers=2),
            DataLoader(test_dataset, batch_size, shuffle=is_shuffle, num_workers=2))


def label2idx(label):
    dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'J': 18, 'K': 19,
            'L': 20, 'M': 21, 'N': 22, 'P': 23, 'Q': 24, 'R': 25, 'S': 26, 'T': 27, 'U': 28, 'V': 29,
            'W': 30, 'X': 31, 'Y': 32, 'Z': 33, 'hu': 34, 'jing': '35'}

    return [dict[label] for label in label]


def idx2label(idx):
    dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
            10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'J', 19: 'K',
            20: 'L', 21: 'M', 22: 'N', 23: 'P', 24: 'Q', 25: 'R', 26: 'S', 27: 'T', 28: 'U', 29: 'V',
            30: 'W', 31: 'X', 32: 'Y', 33: 'Z', 34: 'hu', 35: 'jing'}
    if type(idx) == list:
        return [dict[index] for index in idx]
    elif type(idx) == int:
        return dict[idx]
    elif type(idx) == torch.Tensor:
        return [dict[index] for index in idx.tolist()]
    else:
        raise TypeError


# 反转字典
def dict_reversed(dict):
    dict_reverse = {}
    for key in dict:
        dict_reverse[dict[key]] = key
    return dict_reverse


def timer(f):
    """计时函数运行时间"""

    def timeit(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        print('运行时间:{:.5f}秒'.format(time.time() - start))

        return ret

    return timeit


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # @save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            img = img.permute(1, 2, 0)
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def try_gpu(data, label):
    if torch.cuda.is_available():
        data, label = data.cuda(), label.cuda()
    return data, label


ones = torch.ones
zeros = torch.zeros
tensor = torch.tensor
arange = torch.arange
meshgrid = torch.meshgrid
sin = torch.sin
sinh = torch.sinh
cos = torch.cos
cosh = torch.cosh
tanh = torch.tanh
linspace = torch.linspace
exp = torch.exp
log = torch.log
normal = torch.normal
rand = torch.rand
matmul = torch.matmul
int32 = torch.int32
float32 = torch.float32
concat = torch.cat
stack = torch.stack
abs = torch.abs
eye = torch.eye
numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)

if __name__ == '__main__':
    train, val, test = load_datasets(batch_size=4)
    print('len1:', len(train), len(val), len(test))
    train1 = next(iter(train))[1]
    val1 = next(iter(train))[1]
    test1 = next(iter(train))[1]
    print(train1)
    print(val1)
    print(test1)
    print(idx2label(train1))
    print(idx2label(val1))
    print(idx2label(test1))
