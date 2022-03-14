#!/usr/bin/env python
# coding: UTF-8 
# Created by Cmoon-cyl

import numpy as np
import time


class Timer:
    """计时"""

    def __init__(self):
        self.times = []
        self.start()
        self.tik = None

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]


class Board:
    """0代表无棋子,1代表有皇后"""

    def __init__(self, n):
        self.board = np.zeros((n, n))
        self.n = n

    def set(self, row, col):
        self.board[row][col] = 1

    def reset(self):
        self.board = np.zeros((self.n, self.n))


class NQueens:
    """n为n皇后问题"""
    def __init__(self, n):
        self.board = Board(n)
        self.n = n
        self.timer = Timer()
        self.result = []
        self.answers = []

    def solve(self):
        self.timer.start()
        self.recursion(self.n, [], [[], [], []])
        self.show()
        print('共有{}种可能的布局,运行时间{}ms'.format(len(self.answers), self.timer.stop() * 1000))

    def recursion(self, n, result, pre_forbid):
        """递归求解"""
        if len(result) == n:  # 递归退出条件
            self.result.append(result)
            return
        left, right, down = pre_forbid  # 放置棋子的行下一行的左下右三个格子不能放置
        left, right = [i - 1 for i in left], [i + 1 for i in right]  # 左右格每行更新
        forbid = left + right + down
        for pos in range(n):
            if pos in forbid:
                continue
            else:
                self.recursion(n, result + [pos], [left + [pos], right + [pos], down + [pos]])

    def show(self):
        """转换结果供显示"""
        for num, answer in zip(range(len(self.result)), self.result):
            for row, col in zip(range(self.n), answer):
                self.board.set(row, col)
            self.answers.append(self.board.board)
            print('第{}种布局:\n{}'.format(num + 1, self.board.board))
            self.board.reset()


if __name__ == '__main__':
    problem = NQueens(8)
    problem.solve()
