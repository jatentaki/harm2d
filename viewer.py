import argparse, os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)

args = parser.parse_args()

files = [f for f in os.listdir(args.path) if f.endswith('.npy')]

for file in files:
    c = np.load(args.path + os.path.sep + file)
    plt.imshow(c)
    plt.show()
