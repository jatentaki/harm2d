import argparse, os, re
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)

args = parser.parse_args()

regex = re.compile('^\d+$')
def is_nr_dir(name):
    is_nr = regex.match(name) is not None
    is_dir = os.path.isdir(args.path + os.path.sep + name)
    return is_nr and is_dir 
    
dirs = [f for f in os.listdir(args.path) if is_nr_dir(f)]

for dir in dirs:
    dir_p = os.path.join(args.path, dir)
    files = [f for f in os.listdir(dir_p) if f.endswith('.npy')]
    fig, axes = plt.subplots(1, len(files))
    fig.suptitle(dir_p)
    for file, ax in zip(files, axes):
        c = np.load(os.path.join(args.path, dir, file))
        ax.imshow(c)
    plt.show()
