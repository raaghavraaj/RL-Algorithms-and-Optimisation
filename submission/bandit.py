import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# adding command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--instance", help="path of the instance file")
parser.add_argument("--algorithm", help="for one of the 7 algorithms")
parser.add_argument("--randomSeed", help="non-negative integer")
parser.add_argument(
    "--epsilon", help="a real number in [0, 1] for epsilon-greedy methods")
parser.add_argument(
    "--scale", help="a positive real number; relevant for Task 2")
parser.add_argument(
    "--threshold", help="a number in [0, 1]; relevant for Task 4")
parser.add_argument(
    "--horizon", help="a non-negative integer denoting the number of trials/attempts")

# parsing the command line arguments
args = parser.parse_args()

instance = args.instance
algo = args.algorithm
seed = int(args.randomSeed)
eps = float(args.epsilon)
c = float(args.scale)
th = float(args.threshold)
horizon = int(args.horizon)

out = "{}, {}, {}, {}, {}, {}, {}, ".format(
    instance, algo, seed, eps, c, th, horizon)

# Seeding
np.random.seed(seed)


# Class arm
class arm:

    def __init__(self, mean):
        self.p = mean

    def pull(self):
        pl = np.random.uniform(0, 1)
        if pl < self.p:
            return 0
        else:
            return 1


# methods
def epsG(arms, epsilon, T):
    N = len(arms)
    arm_successes = np.zeros(N)
    arm_pulls = np.zeros(N)
    empirical_means = np.zeros(N)
    reward = 0

    for i in range(T):
        toss = np.random.uniform(0, 1)
        ind = 0

        if toss < epsilon:
            ind = np.random.choice(N)
        else:
            ind = np.argmax(empirical_means)
        arm_pull = arms[ind].pull()
        reward += arm_pull
        arm_successes[ind] += arm_pull
        arm_pulls[ind] += 1
        empirical_means[ind] = arm_successes[ind] / arm_pulls[ind]

    ind_max = np.argmax([arm.p for arm in arms])
    regret = arms[ind_max].p * T - reward
    return regret, 0


def UCB(arms, T, scale):
    N = len(arms)
    arm_successes = np.zeros(N)
    arm_pulls = np.zeros(N)
    empirical_means = np.zeros(N)
    reward = 0

    # to begin with the algorithm, we sample all arms once
    for i in range(N):
        arm_pull = arms[i].pull()
        reward += arm_pull
        arm_successes[i] = arm_pull
        arm_pulls[i] += 1
        empirical_means[i] = arm_successes[i] / arm_pulls[i]

    for i in range(N, T):
        ind = np.argmax([(p + np.sqrt(scale * np.log(i)/n))
                        for p, n in zip(empirical_means, arm_pulls)])
        arm_pull = arms[ind].pull()
        reward += arm_pull
        arm_successes[ind] = arm_pull
        arm_pulls[ind] += 1
        empirical_means[ind] = arm_successes[ind] / arm_pulls[ind]

    ind_max = np.argmax([arm.p for arm in arms])
    regret = arms[ind_max].p * T - reward
    return regret, 0


def klBern(x, y):
    # function to calculate the kl-divergence between two samples
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


def ucb_kl(p, d):
    # function to calculate the ucb-kl for the arm
    value = p
    u = 1
    _count_iteration = 0
    while _count_iteration < 100 and u - value > 1e-6:
        _count_iteration += 1
        m = (value + u) / 2.
        if klBern(p, m) > d:
            u = m
        else:
            value = m
    return (value + u) / 2


def kl_UCB(arms, T):
    N = len(arms)
    arm_successes = np.zeros(N)
    arm_pulls = np.zeros(N)
    empirical_means = np.zeros(N)
    reward = 0

    # to begin with the algorithm, we sample all arms once
    for i in range(N):
        arm_pull = arms[i].pull()
        reward += arm_pull
        arm_successes[i] = arm_pull
        arm_pulls[i] += 1
        empirical_means[i] = arm_successes[i] / arm_pulls[i]

    for i in range(N, T):
        ind = np.argmax([ucb_kl(p, (np.log(i) + 3*np.log(np.log(i)))/n)
                        for p, n in zip(empirical_means, arm_pulls)])
        arm_pull = arms[ind].pull()
        reward += arm_pull
        arm_successes[ind] = arm_pull
        arm_pulls[ind] += 1
        empirical_means[ind] = arm_successes[ind] / arm_pulls[ind]

    ind_max = np.argmax([arm.p for arm in arms])
    regret = arms[ind_max].p * T - reward
    return regret, 0


# class of bandit, with several methods
class Bandit:
    # arms = []

    def __init__(self, instance):
        with open(instance) as f:
            self.arms = [arm(float(i)) for i in f.read().splitlines()]

    def runAlgo(self, algorithm):

        # prints the cmd line arguments, and the REGRET and HIGH
        if algorithm == 'epsilon-greedy-t1':
            reg, highs = epsG(self.arms, eps, horizon)

        elif algorithm == 'ucb-t1':
            reg, highs = UCB(self.arms, horizon, c)

        elif algorithm == 'kl-ucb-t1':
            reg, highs = kl_UCB(self.arms, horizon)

        print(out + "{}, {}\n".format(reg, highs))


test = Bandit(instance)
test.runAlgo(algo)
