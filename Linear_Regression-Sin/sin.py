#!/usr/bin/env python
# coding: UTF-8 
# Created by Cmoon-Cyl

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import matplotlib.pyplot as plt
import time


class Utils:
    """工具函数类"""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def synthetic_data(self, num=1000):
        """生成数据"""
        X = torch.linspace(-2 * np.pi, 2 * np.pi, steps=num, dtype=torch.float32).reshape(-1, 1).to(self.device)
        Y = torch.sin(X)
        return X, Y

    def load_array(self, data_arrays, batch_size, is_train=True):
        """生成数据集"""
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    @staticmethod
    def timer(f):
        """计时函数运行时间"""
        def timeit(*args, **kwargs):
            start = time.time()
            ret = f(*args, **kwargs)
            print('运行时间:{:.5f}秒'.format(time.time() - start))

            return ret

        return timeit


class Trainer:
    def __init__(self):
        self.utils = Utils()
        self.batch_size = 256
        self.lr = 8e-3
        self.wc = 1e-4
        self.features, self.labels = self.utils.synthetic_data(5000)
        self.data_iter = self.utils.load_array((self.features, self.labels), self.batch_size)
        self.net1 = nn.Sequential(nn.Linear(1, 10), nn.ReLU(),
                                  nn.Linear(10, 128), nn.ReLU(),
                                  nn.Linear(128, 10), nn.ReLU(),
                                  nn.Linear(10, 1)).to(self.utils.device)
        self.net2 = nn.Sequential(nn.Linear(1, 10), nn.Sigmoid(),
                                  nn.Linear(10, 128), nn.Sigmoid(),
                                  nn.Linear(128, 10), nn.Sigmoid(),
                                  nn.Linear(10, 1)).to(self.utils.device)
        self.optim1 = torch.optim.Adam(self.net1.parameters(), lr=self.lr, weight_decay=self.wc)
        self.optim2 = torch.optim.Adam(self.net2.parameters(), lr=self.lr, weight_decay=self.wc)
        self.scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim1, 500)
        self.scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim2, 500)
        self.loss = nn.MSELoss()
        self.train_loss1 = []
        self.train_loss2 = []

    @Utils.timer
    def train(self, epochs=200):
        num_epochs = epochs
        for index, net, optim, scheduler in zip([1, 2], [self.net1, self.net2], [self.optim1, self.optim2],
                                                [self.scheduler1, self.scheduler2]):
            train_loss = []
            if index == 1:
                print('ReLU:')
            else:
                print('Sigmoid:')
            for epoch in range(num_epochs):
                for X, y in self.data_iter:
                    l = self.loss(net(X), y)
                    optim.zero_grad()
                    l.backward()
                    optim.step()
                scheduler.step()
                train_loss.append(l)
                if (epoch + 1) % int((epochs) / 5) == 0:
                    l = self.loss(net(self.features), self.labels)
                    print(f"Epoch:{epoch + 1}, MSE:{l}, "
                          f"Device:{l.device}, lr:{scheduler.get_last_lr()[0]}")
            if index == 1:
                self.train_loss1 = train_loss
            else:
                self.train_loss2 = train_loss

    def plot_loss(self, loss):
        activate = 'ReLU' if loss == self.train_loss1 else 'Sigmoid'
        plt.plot(range(0, len(loss)), loss, label="loss")
        plt.title(f"{activate} Train Loss")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.show()

    def plot_result(self):
        predict1 = self.net1(self.features)
        predict2 = self.net2(self.features)
        plt.plot(self.features.to('cpu'), self.labels.to('cpu'), label="Ground Truth")
        plt.plot(self.features.to('cpu'), predict1.detach().to('cpu').numpy(), label="ReLU")
        plt.plot(self.features.to('cpu'), predict2.detach().to('cpu').numpy(), label="Sigmoid")
        plt.title("Result")
        plt.xlabel("X")
        plt.ylabel("Sin(X)")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train(200)
    trainer.plot_loss(trainer.train_loss1)
    trainer.plot_loss(trainer.train_loss2)
    trainer.plot_result()
