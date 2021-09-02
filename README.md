# RL-Algorithms-and-Optimisation
The repository contains the beginner Reinforcement Learning algorithms dealing with the Exploration vs Exploitation dilemma, which are:
* Epsilon-Greedy Algorithm
* UCB Algorithm
* KL-UCB Algorithm
* Thompson Sampling Algorithm

# Running and executing the algorithms
All of the classes, algorithms and useful functions are implemented in the file ```submission/bandit.py```

```bandit.py``` takes 7 command line arguments:
* ```--instance```: the path of the the bandit instance
* ```--algorithm```: the algorithm out of ```epsilon-greedy-t1```, ```ucb-t1```, ```kl-ucb-t1```, ```thompson-sampling-t1```, ```ucb-t2```, ```alg-t3```, ```alg-t4```
* ```--randomSeed```: a non-negative integer to make the process deterministic
* ```--epsilon```: a non-negative real number in \[0, 1\]; relevant to the Epsilon Greedy methods
* ```--scale```: a positive real number, denoting the confidence value in the UCB algorithm, _c_
* ```--threshold```: a real number in \[0, 1\]; relevant to the Task 4
* ```--horizon```: the number of pulls
