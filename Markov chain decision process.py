# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:18:21 2024

@author: Haizhou Cui
"""
import numpy as np
import matplotlib.pyplot as plt

"我们执行一个马尔可夫奖励过程MRP"
np.random.seed(0)
# 定义状态转移矩阵P
P = [
     [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
     [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
     [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
     [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
     ]
P = np.array(P) # 要把P化作数组（矩阵）

rewards = [-1, -2, -2, 10, 1, 0] # 定义奖励函数，e.g., rewards矩阵中的第三个元素表示如果我们处于s3的状态会获得的奖励为-2
gamma = 0.5 # 定义折扣因子
# 给定一条序列，计算从某个索引（起始状态）开始到序列最后（终止状态）得到的回报
def compute_return(start_index, chain, gamma):
    G = 0 
    for i in reversed(range(start_index, len(chain))):
        G = gamma * G + rewards[chain[i]-1]
    return G

# 一个状态序列, s1-s2-s3-s6
chain = [1, 2, 3, 6]
start_index = 0 
G = compute_return(start_index, chain, gamma)
print("根据本序列计算得到回报为：%s" % G)

""" 接下来编写代码来实现求解价值函数（即各状态的价值是多少）的解析解的方法，并据此计算该
马尔科夫链奖励过程中所有状态的价值 """
def compute(P, rewards, gamma, states_num):
    """ 利用贝尔曼方程的矩阵形式计算解析解，states_num是MPR的状态数 """
    rewards = np.array(rewards).reshape((-1,1))
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
    return value

V = compute(P, rewards, gamma, 6)
print("MRP中每个状态价值分别为\n", V)

"我们执行一个马尔可夫决策过程MDP"
S = ["s1", "s2", "s3", "s4", "s5"] # 状态集合
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"] # 动作集合
# 状态转移函数
P= {
    "s1-保持s1-s1" : 1.0,  "s1-前往s2-s2" : 1.0,
    "s2-前往s1-s1" : 1.0,  "s2-前往s3-s3" : 1.0,
    "s3-前往s4-s4" : 1.0,  "s3-前往s5-s5" : 1.0,
    "s4-前往s5-s5" : 1.0,  "s4-概率前往-s2" : 0.2,
    "s4-概率前往-s3" : 0.4, "s4-概率前往-s4" : 0.4,
    }
# 奖励函数
R = {
     "s1-保持s1" : -1, "s1-前往s2" : 0,
     "s2-前往s1" : -1, "s2-前往s3" : -2,
     "s3-前往s4" : -2, "s3-前往s5" : 0,
     "s4-前往s5" : 10, "s4-概率前往" : 1,
     }
gamma = 0.5 # 折扣因子
MDP = (S, A, P, R, gamma) # 一个叫MDP的元组含有马尔可夫决策过程的各要素

# 策略1, 随机策略
Pi_1 = {
     "s1-保持s1" : 0.5, "s1-前往s2" : 0.5,
     "s2-前往s1" : 0.5, "s2-前往s3" : 0.5,
     "s3-前往s4" : 0.5, "s3-前往s5" : 0.5,
     "s4-前往s5" : 0.5, "s4-概率前往" : 0.5,
     }
# 策略2
Pi_2 = {
     "s1-保持s1" : 0.6, "s1-前往s2" : 0.4,
     "s2-前往s1" : 0.3, "s2-前往s3" : 0.7,
     "s3-前往s4" : 0.5, "s3-前往s5" : 0.5,
     "s4-前往s5" : 0.1, "s4-概率前往" : 0.9,
     }
# 把输入的两个字符串通过“-”连接，便于使用上述定义的P,R变量
def join(str1, str2):
    return str1 + '-' + str2

# 计算状态转移矩阵和奖励向量
def mdp_to_mrp(S, A, P, R, Pi):
    states_num = len(S)
    P_pi = np.zeros((states_num, states_num)) # 维护的P_pi用于储存转化成的MRP的状态转移概率
    R_pi = np.zeros((states_num, 1)) # 维护的R_pi用于储存转化成的MRP的各状态奖励值
    
    state_index = {s: i for i, s in enumerate(S)}
    
    for s in S:
        s_idx = state_index[s]
        for a in A:
            prob_action = Pi.get(join(s, a), 0)
            if prob_action > 0:
                for s_next in S:
                    key = join(join(s, a), s_next)
                    P_pi[s_idx, state_index[s_next]] += prob_action * P.get(key, 0)
                R_pi[s_idx] += prob_action * R.get(join(s, a), 0)
                
    return P_pi, R_pi

# 计算P_from_mdp_to_mrp和R_from_mdp_to_mrp
P_from_mdp_to_mrp, R_from_mdp_to_mrp = mdp_to_mrp(S, A, P, R, Pi_1)

""""我们将上述的的MDP转换成对应的MRP, 已方便我们通过前面编写的求解各个状态解析解的方法来求解MRP的
状态价值（也就是MRP转化为MDP后的各个状态的价值）"""
gamma = 0.5
# 转化后的MRP的状态转移矩阵
P_from_mdp_to_mrp = [
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.1, 0.2, 0.2, 0.5],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    ]
P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]

V = compute(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, 5)
print("MDP中每个状态价值分别为\n", V)

# 根据我们求得的，由MDP实例转化为MRP的各个状态的价值，返回到原 MDP 的动作价值Q_pi(s,a)
def compute_action_value(S, A, P, R, gamma, V):
    state_index = {s: i for i, s in enumerate(S)}
    Q = {} # 维护的一个字典Q用于储存原MDP的各动作价值
    
    for s in S:
        for a in A:
            q_value = R.get(join(s, a), 0)
            for s_next in S:
                key = join(join(s, a), s_next)
                if key in P:
                    q_value += gamma * P[key] * V[state_index[s_next]]
            Q[join(s, a)] = q_value
    
    return Q

# 计算动作价值 Q_pi(s, a)
Q_pi = compute_action_value(S, A, P, R, gamma, V)

print("MDP 中每个动作价值分别为:")
for sa in Q_pi:
    print(f"Q_pi({sa}) = {Q_pi[sa]}") # 原MDP的各动作价值储存在一个字典中，而不是一个数组或者列表中

"使用蒙特卡洛方法求解MDP实例各个状态价值V_pi(s),以我们定义的MDP实例执行Pi_1策略为例"
def sample(MDP, Pi, timestep_max, number):
    """定义采样函数，输入相应的MDP（即我们需要一个代表我们要模拟的MDP的元组MDP=(S, A, P, R, gamma))，
    ，策略Pi,限制最长时间步timestep_max, 总共采样序列数number"""
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)] # 随机选择一个除s5以外的状态s作为起点，因为s5是终止状态，不能作为起点
        # 当前状态为终止状态或者时间步长太长时，一次采样结束
        while s != "s5" and timestep <= timestep_max: 
            timestep += 1
            rand, temp = np.random.rand(), 0
            # 在状态s下根据策略选择动作
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if temp > rand:
                    a = a_opt 
                    r = R.get(join(s,a), 0)
                    break
            rand, temp = np.random.rand(), 0 
            # 根据状态转移概率得到下一个状态s_next
            for s_opt in S:
                temp += P.get(join(join(s,a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next)) # 把(s, a, r, s_next)元组放入序列中
            s = s_next # s_next变成当前状态，开始接下来的循环
        episodes.append(episode)
    return episodes

# 采样5次，每个序列最长不超过20步
episodes = sample(MDP, Pi_1, 20, 5)
print('第一条序列\n', episodes[0])
print('第二条序列\n', episodes[1])
print('第五条序列\n', episodes[4])

# 定义MC函数，使用MC函数可以基于上述定义的sample函数蒙特靠罗模拟出的若干条状态序列计算出MDP中各状态的价值V_pi(s)
def MC(episodes, V, N, gamma):
    for episode in episodes:
        G = 0
        for i in range(len(episode)-1, -1, -1): #一个序列从后往前计算
            (s, a, r, s_next) = episode[i]
            G = r + gamma * G
            N[s] = N[s] + 1
            V[s] = V[s] + (G - V[s])/N[s]
        
timestep_max = 20
# 采样1000次（以我们在上面给出的MDP实例为基础进行模拟采样），即产生1000个状态转移序列，可以自行修改
episodes = sample(MDP,Pi_1,timestep_max, 1000)
gamma = 0.5 
V = {"s1":0, "s2":0, "s3":0, "s4":0, "s5":0}
N = {"s1":0, "s2":0, "s3":0, "s4":0, "s5":0}
MC(episodes, V, N, gamma)
print("使用蒙特卡洛方法计算MDP的状态价值\n", V)
V = np.array([V[s] for s in S]).reshape(-1, 1) # 将我们用蒙特卡洛方法计算出的MDP的个状态价值储存的字典转化为数组，该数组依次排列s1-s5的状态价值计算结果
#基于我们蒙特卡洛方法得到的各状态价值，采用我们定义的compute_action_value来求解该MDP的各动作价值 Q_pi(s, a)
Q_pi = compute_action_value(S, A, P, R, gamma, V)
print("MDP 中每个动作价值分别为:")
for sa in Q_pi:
    print(f"Q_pi({sa}) = {Q_pi[sa]}") # 原MDP的各动作价值储存在一个字典中，而不是一个数组或者列表中

"""我们编写一个基于蒙特卡洛算法来近似估计针对我们前面给的的MDP，如果执行我们提出的Pi_1
和Pi_2，状态动作对（s4,概率前往）的占用度量在这两个策略下分别是多少"""

"""我们定义一个名叫occupancy的函数，通过输入我们通过sample函数模拟出的采样结果episodes(含有若干条状态转移序列);
我们要测算占用度量的状态动作对s,a; 允许的最大步长timestep_max; 折现因子gamma, 就可以直接返回我们
采用蒙特卡洛模拟对s,a状态动作对占用度量的近似求解""" 
def occupancy (episodes, s, a, timestep_max, gamma):
    ''' 计算状态动作对(s,a)出现的频率，以此来估算策略的占用度量 '''
    rho = 0 
    total_times = np.zeros(timestep_max) # 维护该数组用于记录每个时间步t各被经历过几次
    occur_times = np.zeros(timestep_max) # 维护该数组用于记录我们要测算占用度量的的状态动作对(s,a)在各时间步上被出现了多少次，即(s_t, a_t)=(s,a)的次数
    for episode in episodes:
        for i in range(len(episode)):
            (s_opt, a_opt, r, s_next) = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times[i] += 1
    for i in reversed(range(timestep_max)):
        if total_times[i]:
                rho += gamma**i * occur_times[i] / total_times[i]
    return (1 - gamma) * rho
    
gamma = 0.5
timestep_max = 1000
# 对Pi_1策略采样1000次，产生1000条状态转移序列
episodes_1 = sample(MDP, Pi_1, timestep_max, 1000)
# 对Pi_2策略采样1000次，产生1000条状态转移序列
episodes_2 = sample(MDP, Pi_2, timestep_max, 1000)
# 分别计算Pi_1和Pi_2策略下，状态动作对（s4,概率前往）的占用度量rho_1和rho_2
rho_1 =  occupancy(episodes_1, "s4", "概率前往", timestep_max, gamma)
rho_2 =  occupancy(episodes_2, "s4", "概率前往", timestep_max, gamma)
print(f"该MDP下状态动作对（s4,概率前往）在策略Pi_1和策略Pi_2下的占用度量分别为{rho_1}，{rho_2}")
# 通过上述的结果可以发现，不同策略对于同一个状态动作对的占用度量是不一样的

"""要计算并存储一个MDP在某个策略Pi下所有状态动作对的占用度量, 我们可以通过维
护一个字典来储存这个MDP在该策略Pi下所有状态动作对的占用度量。
以及一个数组也来储存这个MDP在该策略Pi下所有状态动作对的占用度量（数组的行对应状态，列对应动作）。
如果某一个状态动作对不存在在该MDP中，该状态动作对的占用度量为0储存在该数组或字典中"""
# 创建储存状态动作对占用度量字典
def compute_occupancy_dict(MDP, Pi, episodes, timestep_max, gamma):
    S, A, _, _, _ = MDP #这是因为我们定义的compute_occupancy_dict的函数在运算过程中只会使用到S和A两个参数，但是MDP这个元组在拆包时会拆出5个元素，我们只用前两个，即S，A，因此后面拆包出的元素用_表示即可
    occupancy_dict = {}
    for s in S:
        for a in A:
            rho = occupancy(episodes, s, a, timestep_max, gamma)
            occupancy_dict[join(s, a)] = rho
    return occupancy_dict

# 创建储存状态动作对占用度量数组
def compute_occupancy_array(MDP, Pi, episodes, timestep_max, gamma):
    S, A, _, _, _ = MDP
    occupancy_array = np.zeros((len(S), len(A)))
    for i, s in enumerate(S):
        for j, a in enumerate(A):
            rho = occupancy(episodes, s, a, timestep_max, gamma)
            occupancy_array[i, j] = rho
    return occupancy_array

episodes_1 = sample(MDP, Pi_1, timestep_max, 1000)
# 计算状态动作对占用度量字典
occupancy_dict_1 = compute_occupancy_dict(MDP, Pi_1, episodes_1, timestep_max, gamma)
print("Pi_1策略下状态动作对占用度量字典:\n", occupancy_dict_1)
# 计算状态动作对占用度量数组
occupancy_array_1 = compute_occupancy_array(MDP, Pi_1, episodes_1, timestep_max, gamma)
print("Pi_1策略下状态动作对占用度量数组:\n", occupancy_array_1)