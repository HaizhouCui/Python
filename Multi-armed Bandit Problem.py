# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:28:22 2024

@author: Haizhou Cui
"""
import numpy as np
import matplotlib.pyplot as plt
class BernoulliBandit: 
    """ 伯努利多臂老虎机，输入K表示拉杆个数 """
    def __init__ (self, K):
        self.probs = np.random.uniform(size=K) # 随机生成K个0~1的数，作为拉动每根拉杆的获奖概率
        self.best_idx = np.argmax(self.probs) # 获奖概率最大的拉杆编号
        self.best_prob = self.probs[self.best_idx] #最大的获奖概率
        self.K=K
    
    def step(self, k):
        # 当玩家选择了k号拉杆后，根据拉动该老虎机的k号拉杆获得奖励的概率，返回是否获奖的结果，
        # 获奖返回1，不获奖返回0
        if np.random.rand() < self.probs[k]:
            return 1
        else: 
            return 0

np.random.seed(1) # 设定随机数种子，使实验具有可重复性
K=10
bandit_10_arm = BernoulliBandit(K) # 这里我们模拟生成了一个10拉杆的多臂老虎机
print("随机生成了一个%d臂伯努利老虎机" % K)
print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" 
      % (bandit_10_arm.best_idx, bandit_10_arm.best_prob))
bandit_10_arm.step(7) # 表示我们尝试拉动第七根拉杆，看看是否获奖

class Solver:
    """ 多臂老虎机算法基本框架 """
    def __init__ (self, bandit):
        self.bandit = bandit 
        self.counts = np.zeros(self.bandit.K) # 每根拉杆的尝试次数
        self.regret = 0. # 当前步的累积懊悔
        self.actions = [] # 维护一个列表，记录每一步的动作
        self.regrets = [] # 维护一个列表，记录每一步的累积懊悔
    
    def update_regret(self, k):
        # 计算累积懊悔并，k为本次动作选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)
        
    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆，由每个具体策略实现
        raise NotImplementedError()
        
    def run(self, num_steps): 
        # 运行一定次数，num_steps为总运行次数
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1 
            self.actions.append(k)
            self.update_regret(k)

class EpsilonGreedy(Solver):
    """ epsilon贪婪算法，继承Solver类 """
    def __init__  (self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        # 初始化拉动所有拉杆的期望奖励估计
        self.estimates = np.array([init_prob] * self.bandit.K)
        
    def run_one_step(self):
        """ 实现了如每轮拉杆时，有(1-epsilon)的概率选择历史奖励期望（均值）最高的拉杆， epsilon有
        的概率选择随机一根拉杆 """
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K) # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates) # 选择期望奖励估计最大的拉杆
        r = self.bandit.step(k) # 根据先前定义的bandit类的程序，得到本次拉杆动作的结果与奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r-self.estimates[k])
        return k
    
def plot_results(solvers, solver_names):
    """ 生成累积懊悔随时间变化的图像。输入solvers是一个列表，列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表，存储每个策略的名名称。也就是说，在我们使用我们自定义的
    plots_results函数时, 我们要输入两个参数，即solvers和solver_names，这两个参数都是列表。
    换句话说我们在使用plot_results()函数时在括号内输入的是两个列表 """
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

np.random.seed(1)
"构建一个解决我们模拟的10臂伯努利老虎机的epsilon贪婪算法的实例，取ε=0.01"
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01) # ε-贪婪算法解决老虎机问题的实例的构建
epsilon_greedy_solver.run(5000) # ε-贪婪算法在5000轮游戏中进行
print('ε-贪婪算法在5000轮完成后的累积懊悔为: ', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])

np.random.seed(0)
"""我们想要同时考察ε=0.0001, 0.01, 0.1, 0.25和0.5时ε-贪婪算法求解该10臂老虎机问题，于是我们把
ε=0.0001, 0.01, 0.1, 0.25和0.5下的EpsilonGreedy的实例放在一个列表里面，同时把这些实例的名字以
字符串的形式写在一个列表中。这是因为我们希望将这些不同ε取值的实例横向比较，并且把他们的
（轮次-懊悔）图画在一起"""
epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
"""生成一个包含若干EpsilonGreedy实例的列表（即把
ε=0.0001, 0.01, 0.1, 0.25和0.5下的EpsilonGreedy的实例放在一个列表里面）"""
epsilon_greedy_solver_list=[EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons]
"""生成一个包含若干EpsilonGreedy实例的命名字符串的列表（即把=0.0001, 0.01, 0.1, 0.25
和0.5下的EpsilonGreedy的实例的名字以字符串的形式写在一个列表中"""
epsilon_greedy_solver_names=["epsilon={}".format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)
plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

class DecayingEpsilonGreedy(Solver):
    """ epsilon值随时间反比例衰减的epsilon-贪婪算法，继承Solver类 """
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count=0
    
    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1/self.total_count: # epsilon值是轮次的反比
        # 即epsilon等于1比上轮次数
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        
        return k
    
np.random.seed(1)
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])

class UCB(Solver):
    """UCB算法，继承Solver类"""
    def __init__(self, bandit, coef, init__prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init__prob] * self.bandit.K)
        self.coef = coef
        
    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / 
                 (2 * (self.counts+ 1))) # 计算上置信界
        k = np. argmax(ucb) # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

np.random.seed(1)
coef = 1 # 控制不确定性比重的系数
UCB_solver = UCB(bandit_10_arm, coef) # 用上置信界UCB算法解决我们定义的10臂伯努利老虎机问题
UCB_solver.run(5000)
print('上置信界算法的累积懊悔为：', UCB_solver.regret)
plot_results([UCB_solver], ["UCB"])

class ThompsonSampling(Solver):
    """ 汤普森采样算法，继承Solver类 """
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K) # 列表, 表示每根拉杆奖励为1的次数, 作为α的取值
        self._b = np.ones(self.bandit.K) # 列表, 表示每根拉杆奖励为0的次数，作为β的取值
        
    def run_one_step(self):
        samples = np.random.beta(self._a, self._b) # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples) # 选出采样奖励最大的拉杆作为本轮策略
        r = self.bandit.step(k)
        
        self._a[k] += r # 更新Beta分布的第一个参数α
        self._b[k] += (1-r) # 更新Beta分布的第二个参数β
        return k
    
"""我们运用汤普森采样算法解决我们定义的10臂老虎机问题"""
thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
plot_results([thompson_sampling_solver], ["ThompsonSampling"])
        
        