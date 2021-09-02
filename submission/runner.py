import os

algos = ['epsilon-greedy-t1', 'ucb-t1', 'kl-ucb-t1',
         'thompson-sampling-t1', 'ucb-t2', 'alg-t3', 'alg-t4']

horizons = [100, 400, 1600, 6400, 25600, 102400]

# Task 1:
for i in range(1, 4):
    instance = "../instances/instances-task1/i-{}.txt".format(i)
    for algo in algos[0:3]:
        for horizon in horizons:
            for seed in range(50):
                cmd = "python3 bandit.py --instance {} --algorithm {} --randomSeed {} --epsilon 0.02 --scale 2 --threshold 0 --horizon {} >> outputData.txt".format(
                    instance, algo, seed, horizon)
                os.system(cmd)

# Task 2:
for i in range(1, 6):
    instance = "../instances/instances-task2/i-{}.txt".format(i)
    for scale in range(1, 16):
        for seed in range(50):
            cmd = "python3 bandit.py --instance {} --algorithm ucb-t2 --randomSeed {} --epsilon 0.02 --scale {} --threshold 0 --horizon 10000 >> outputData.txt".format(
                instance, seed, scale)
            os.system(cmd)

# Task 3:
for i in range(1, 3):
    instance = "../instances/instances-task3/i-{}.txt".format(i)
    for horizon in horizons:
        for seed in range(50):
            cmd = "python3 bandit.py --instance {} --algorithm alg-t3 --randomSeed {} --epsilon 0.02 --scale 2 --threshold 0 --horizon {} >> outputData.txt".format(
                instance, seed, horizon)
            os.system(cmd)

# Task 4:
for i in range(1, 3):
    instance = "../instances/instances-task4/i-{}.txt".format(i)
    for threshold in [0.2, 0.6]:
        for horizon in horizons:
            for seed in range(50):
                cmd = "python3 bandit.py --instance {} --algorithm alg-t4 --randomSeed {} --epsilon 0.02 --scale 2 --threshold {} --horizon {} >> outputData.txt".format(
                    instance, seed, threshold, horizon)
                os.system(cmd)
