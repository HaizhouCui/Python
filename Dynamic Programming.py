# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:52:23 2024

@author: Haizhou Cui
"""

import numpy as np
import matplotlib.pyplot as plt
""" 使用策略迭代和价值迭代来求解悬崖漫步这个环境中的最优策略 """
# 搭建一个悬崖漫步环境
import copy
class CliffWalkingEnv: 
    """ 悬崖漫步环境 """
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol # 定义网格世界的列
        self.nrow = nrow # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()
        
    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作，change[0]:上, change[1]:下, change[2]:左, change[3]:右。 坐标系原点(0,0)
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态，因为无法继续交互，任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1 and next_x > 0: # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P

env = CliffWalkingEnv()

# 打印部分状态转移矩阵示例
for state in range(env.nrow * env.ncol):
    print(f"State {state}:")
    for action in range(4):
        transitions = env.P[state][action]
        print(f"  Action {action}:")
        for prob, next_state, reward, done in transitions:
            print(f"    -> (prob: {prob}, next_state: {next_state}, reward: {reward}, done: {done})")                            