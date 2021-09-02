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
instance = parser.instance
algo = parser.algorithm
seed = parser.
if __name__ == "__main__":
