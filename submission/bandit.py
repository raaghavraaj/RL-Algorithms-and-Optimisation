import os
import random
import subprocess
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
            ind = random.randint(0, N - 1)
        else:
            ind = np.argmax(empirical_means)

        reward += arms[ind].pull()
        arm_successes[ind] += arms[ind].pull()
        arm_pulls[ind] += 1
        empirical_means[ind] = arm_successes[ind] / arm_pulls[ind]

    ind_max = np.argmax([arm.p for arm in arms])
    regret = arms[ind_max].p * T - reward
    return regret, 0


def ucb(arms, T):
    N = len(arms)


# class of bandit, with several methods
class Bandit:
    # arms = []

    def __init__(self, instance):
        with open(instance) as f:
            self.arms = [arm(float(i)) for i in f.read().splitlines()]

    def runAlgo(self, algorithm):
        # prints the cmd line arguments, and the REGRET and HIGH
        out = "{}, {}, {}, {}, {}, {}, {}, ".format(
            instance, algo, seed, eps, c, th, horizon)

        if algorithm == 'epsilon-greedy-t1':
            reg, highs = epsG(self.arms, eps, horizon)
            print(out + "{}, {}".format(reg, highs))

        elif algorithm == 'ucb-t1':
            reg, highs = ucb(self.arms, horizon)


test = Bandit(instance)
test.runAlgo(algo)
