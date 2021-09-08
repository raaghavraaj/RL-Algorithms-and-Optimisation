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

and prints the 7 arguments along with the REGRET and HIGHS calculated in a comma-separated form.

The script can be run using the following command using the appropriate and customised parameters for the conditional arguments:
```
~RL-Algorithms-and-Optimisation $ python3 bandit.py --instance {path} --algorithm {algo} --randomSeed {seed} --epsilon {eps} --scale {c} --threshold {th} --horizon {hz}
```
For example
```
~RL-Algorithms-and-Optimisation $ python3 bandit.py --instance ../instances/instances-task1/i-2.txt --algorithm ucb-t1 --randomSeed 499 --epsilon 0.02 --scale 2 --threshold 0 --horizon 27
```
Output
```
../instances/instances-task1/i-2.txt, ucb-t1, 499, 0.02, 2.0, 0.0, 27, 6.5, 0
```
The ```submissions``` directory contains a script ```runner.py``` to run all the tasks and prints data. Run the script and pipe the output to a file (there will be 9000+ lines printed). The complete task would take almost 4-6 hours to run.
The bottom line is an output example.
