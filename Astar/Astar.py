#!/usr/bin/env python
# coding: UTF-8
# Created by Cmoon-cyl

import sys, pygame
import time

color = {'grey': (211, 211, 211), 'red': (255, 0, 0), 'black': (0, 0, 0), 'green': (0, 255, 120),
         'orange': (255, 165, 0), 'bg': (240, 255, 255)}


class Node:
    """节点类"""

    def __init__(self, point, parent, g, h):
        super(Node, self).__init__()
        self.point = point  # 节点坐标
        self.parent = parent  # 父节点对象地址
        self.g = g  # 起点至该节点的确定距离
        self._h = h  # 节点到目标点曼哈顿距离
        self.f = self.g + self._h

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, value):
        """对h赋值时自动更新f"""
        self._h = value
        self.f = self._h + self.g


class Board:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((660, 660))
        self.fpsClock = pygame.time.Clock()
        pygame.display.set_caption('Astar Algorithm Demo')
        self.width = 30
        self.obs = []

    def init_grid(self):
        """初始化栅格"""
        for a in range(0, 660, self.width):
            pygame.draw.line(self.screen, color['grey'], [a, 0], [a, 660])
            pygame.draw.line(self.screen, color['grey'], [0, a], [660, a])

    def judge_in_grid(self, p):
        p0 = int(p[0] / 30)
        p1 = int(p[1] / 30)
        rect = ((p0 * 30, p1 * 30), (30, 30))
        return rect

    def init_obstacles(self):
        """初始化障碍"""
        self.obs = []
        self.obs.append(pygame.Rect((270, 270), (120, 30)))
        for rect in self.obs:
            pygame.draw.rect(self.screen, color['black'], rect)

    def collides(self, p):
        """检测是否碰撞"""
        for rect in self.obs:
            if rect.collidepoint(p):
                return True
        return False

    def reset(self):
        self.screen.fill(color['bg'])
        self.init_grid()
        self.init_obstacles()

    def text_display(self, text, rect, color, size):
        font = pygame.font.Font(None, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        text_rect.center = (15 + (rect[0][0]), 15 + (rect[0][1]))
        self.screen.blit(text_surface, text_rect)


class Astar:
    def __init__(self):
        self.delay_sec = 0.2  # 延时
        self.board = Board()
        self.openlist = []
        self.closelist = []
        self.start = Node(None, None, 0, 0)
        self.goal = Node(None, None, 0, 0)

    def manhattan(self, node, goal):
        return abs(node[0] - goal[0]) / 3 + abs(node[1] - goal[1]) / 3

    def evaluate(self, node, goal):
        judge_open = False
        judge_close = False
        if self.board.collides(node.point) or node.point[0] < 0 or node.point[1] < 0:
            return
        node.h = self.manhattan(node.point, goal.point)
        for p in self.openlist:
            if p.point == node.point:
                judge_open = True
                index_open = self.openlist.index(p)
        for p in self.closelist:
            if p.point == node.point:
                judge_close = True
                index_close = self.closelist.index(p)
        if judge_close:
            return
        if not judge_open:
            self.openlist.append(node)
            pygame.draw.rect(self.board.screen,
                             (max(0., 255 - float(node.f)),
                              max(0., 255 - float(node.f)), 255),
                             (node.point, (30, 30)))
            self.board.text_display(str(int(node.f)), (node.point, (30, 30)), color['black'], 15)
            pygame.display.flip()
        else:
            if self.openlist[index_open].g > node.g:
                self.openlist[index_open] = node


class Navigator:
    def __init__(self):
        self.algorithm = Astar()
        self.delay = 0.1
        self.state = 'init'
        self.setStart = True
        self.setGoal = True
        self.flag = True
        self.shortcut = []

    def find_shortcut(self):
        self.algorithm.board.reset()
        while True:
            if self.state == 'init':
                self.algorithm.board.fpsClock.tick(10)
            elif self.state == 'Navigating':
                self.flag = True
                while len(self.algorithm.openlist) != 0:
                    for event in pygame.event.get():
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            self.flag = not self.flag  # 单击鼠标可暂停
                    if self.flag:
                        node = self.algorithm.openlist[0]
                        if node.point == self.algorithm.goal.point:
                            pygame.draw.rect(self.algorithm.board.screen, color['orange'], (node.point, (30, 30)))
                            self.algorithm.board.text_display("B", (node.point, (30, 30)), color['black'], 32)
                            self.state = 'goalFound'
                            flag = 1
                            goal_node = node
                            break
                        self.algorithm.closelist.append(node)
                        self.algorithm.openlist.pop(0)
                        pygame.draw.rect(self.algorithm.board.screen, color['red'], (node.point, (30, 30)), width=1)
                        for i in range(8):
                            if i == 0:
                                node_al = Node((node.point[0], node.point[1] - 30), node, node.g + 10, 0)
                            elif i == 1:
                                node_al = Node((node.point[0], node.point[1] + 30), node, node.g + 10, 0)
                            elif i == 2:
                                node_al = Node((node.point[0] - 30, node.point[1]), node, node.g + 10, 0)
                            elif i == 3:
                                node_al = Node((node.point[0] + 30, node.point[1]), node, node.g + 10, 0)
                            elif i == 4:
                                node_al = Node((node.point[0] - 30, node.point[1] - 30), node, node.g + 14, 0)
                            elif i == 5:
                                node_al = Node((node.point[0] + 30, node.point[1] - 30), node, node.g + 14, 0)
                            elif i == 6:
                                node_al = Node((node.point[0] + 30, node.point[1] + 30), node, node.g + 14, 0)
                            elif i == 7:
                                node_al = Node((node.point[0] - 30, node.point[1] + 30), node, node.g + 14, 0)

                            self.algorithm.evaluate(node_al, self.algorithm.goal)
                        for i in range(0, len(self.algorithm.openlist) - 1):
                            for j in range(0, len(self.algorithm.openlist) - 1 - i):
                                data_j = self.algorithm.openlist[j].f
                                data_j1 = self.algorithm.openlist[j + 1].f
                                if data_j > data_j1:
                                    temp = self.algorithm.openlist[j]
                                    self.algorithm.openlist[j] = self.algorithm.openlist[j + 1]
                                    self.algorithm.openlist[j + 1] = temp
                        pygame.display.flip()
                        for i in range(0, len(self.algorithm.openlist)):
                            print('{} ------- g={} h={} f={}'.format(self.algorithm.openlist[i].point,
                                                                     self.algorithm.openlist[i].g,
                                                                     self.algorithm.openlist[i].h,
                                                                     self.algorithm.openlist[i].f))
                        print("------------------------------------------")
                        time.sleep(self.delay)
                    if len(self.algorithm.openlist) == 0:
                        print('False to find the path')

            elif self.state == 'goalFound':
                if flag == 1:
                    print('最短路径长度: {}\n路径为:'.format(goal_node.g))
                    current = goal_node.parent
                    self.shortcut.append(self.algorithm.goal.point)

                while current.parent != None:
                    self.shortcut.append(current.point)
                    pygame.draw.rect(self.algorithm.board.screen, color['green'], (current.point, (30, 30)))
                    self.algorithm.board.text_display(str(current.g), (current.point, (30, 30)), color['black'], 15)
                    current = current.parent
                    flag = 0
                    pygame.display.flip()
                    time.sleep(float(self.delay) / 2)
                    if current.parent is None:
                        self.shortcut.append(self.algorithm.start.point)
                        for point in reversed(self.shortcut):
                            print(point)
                        print('Task completed!\n\n')
                self.algorithm.board.fpsClock.tick(10)
            # 处理鼠标事件
            self.monitor()
            pygame.display.flip()

    def monitor(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE):
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.state == 'init':
                rect = self.algorithm.board.judge_in_grid(event.pos)
                if self.algorithm.board.collides(event.pos) == False:
                    if self.setStart:
                        self.algorithm.start = Node(rect[0], None, 0, 0)
                        pygame.draw.rect(self.algorithm.board.screen, color['orange'], rect)
                        self.algorithm.board.text_display("A", rect, color['black'], 32)
                        self.setStart = False
                        print('Start Point set to:', str(self.algorithm.start.point))
                    elif self.setGoal:
                        if rect[0] != self.algorithm.start.point:
                            self.algorithm.goal = Node(rect[0], None, 0, 0)
                            print("Goal Point set to:", str(self.algorithm.goal.point))
                            self.algorithm.start.h = self.algorithm.manhattan(
                                self.algorithm.goal.point,
                                self.algorithm.start.point)
                            self.algorithm.openlist.append(self.algorithm.start)
                            pygame.draw.rect(self.algorithm.board.screen, color['orange'], rect)
                            self.algorithm.board.text_display("B", rect, color['black'], 32)
                            self.setGoal = False
                            self.state = 'Navigating'

            if event.type == pygame.KEYUP and event.key == pygame.K_r:
                self.state = 'init'
                self.setStart = True
                self.setGoal = True
                self.algorithm.openlist = []
                self.algorithm.closelist = []
                self.shortcut = []
                self.algorithm.board.reset()


if __name__ == '__main__':
    nav = Navigator()
    nav.find_shortcut()
