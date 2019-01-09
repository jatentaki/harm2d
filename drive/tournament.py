import argparse, os, sys, random
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np

# parent directory
sys.path.append('../')
import transforms as tr

# `drive` directory
import loader
from genetic_runner import Experiment

class Tournament:
    def __init__(self, args):
        self.train_loader, self.test_loader = self.build_loaders(args)
        self.args = args
        self.artifacts = args.artifacts
        self.layers = 50

        if not os.path.isdir(self.artifacts):
            os.makedirs(self.artifacts)

    def __enter__(self):
        self.file = open(f'{self.artifacts}/tournament.txt', 'w')
        return self

    def __exit__(self, *exc):
        self.file.close()

    def log(self, msg):
        self.file.write(f'{msg}\n')
        print(f'LOG: {msg}')
        self.file.flush()

    def play(self):
        population = self.initialize_pool(self.args.num_contestants)
        for round in range(self.args.num_rounds):
            self.log(f'Round {round}')
            evaluations = []
            for i, setup in enumerate(population):
                artifacts = f'{self.args.artifacts}/round_{round}_contestant_{i}'
                self.log(f'Evaluating {setup}, artifacts in {artifacts}')
                score = self.evaluate(setup, artifacts)
                setup.score = score
                evaluations.append(score)

            population = self.mate(population, evaluations)

    def crossover(self, mother, father):
        child_bins = mother.bins + father.bins
        return Setup(child_bins, layers=self.layers)

    def mate(self, phenotypes, evaluations):
        evaluations = np.array(evaluations)
        diffs = evaluations - evaluations.min()
        eval_norm = diffs / diffs.sum()

        size = evaluations.size // 2
        mothers = np.random.choice(evaluations.size, size, p=eval_norm)
        fathers = np.random.choice(evaluations.size, size, p=eval_norm)

        m_hist, _ = np.histogram(mothers, bins=evaluations.size, range=(0, size))
        f_hist, _ = np.histogram(fathers, bins=evaluations.size, range=(0, size))       
        hist = m_hist + f_hist

        eval_hist = np.stack([evaluations, hist], axis=1)

        self.log(f'{eval_hist}')

        new_population = []
        for mother, father in zip(mothers, fathers):
            child = self.crossover(phenotypes[mother], phenotypes[father])
            new_population.append(child)

        i = 0
        while len(new_population) < evaluations.size:
            new_population.append(self.random_setup())
            i += 1

        self.log(f'{i} new contestants added')

        return new_population

    def random_setup(self):
        n_bins = 5 * 3
        bins = np.zeros(n_bins, dtype=np.float32)

        for i in range(self.layers):
            choice = random.randint(0, n_bins-1)
            bins[choice] += 1

        return Setup(bins, layers=self.layers)

    def initialize_pool(self, n):
        return [self.random_setup() for _ in range(n)]

    def evaluate(self, setup, artifacts):
        experiment = Experiment(
            setup, self.train_loader, self.test_loader, artifacts, self.args
        )
        try:
            score = experiment.train()
        except Exception as e:
            self.log(f'Error while evaluating {setup}: {e}')
            score = 0.
        
        self.log(f'Evaluated {setup} at {score}')
        return score

    def build_loaders(self, args):
        transform = T.Compose([
            T.CenterCrop(644),
            T.ToTensor()
        ])

        test_global_transform = tr.Lift(T.Pad(40))

        tr_global_transform = [
            tr.RandomRotate(),
            tr.RandomFlip(),
            tr.Lift(T.Pad(40))
        ]
        tr_global_transform = tr.Compose(tr_global_transform)

        train_data = loader.DriveDataset(
            args.data_path, training=True, bloat=args.bloat,
            img_transform=transform, mask_transform=transform,
            label_transform=transform, global_transform=tr_global_transform
        )

        train_loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
        )

        val_data = loader.DriveDataset(
            args.data_path, training=False, img_transform=transform,
            mask_transform=transform, label_transform=transform,
            global_transform=test_global_transform
        )

        test_loader = DataLoader(
            val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
        )

        return train_loader, test_loader

class Setup:
    def __init__(self, bins, layers=100):
        self.layers = layers 
        self.score = None

        bins = np.array(bins, dtype=np.float32)
        bins = np.ceil(bins / bins.sum() * layers).astype(np.int64)

        while bins.sum() > layers:
            ix = random.randint(0, bins.size - 1)
            if bins[ix] > 0:
                bins[ix] -= 1

        self.bins = bins.tolist()
        assert sum(self.bins) == layers

    @property
    def down(self):
        down = [
            self.bins[0:3],
            self.bins[3:6],
            self.bins[6:9]
        ]
        return down

    @property
    def up(self):
        up = [
            self.bins[9:12],
            self.bins[12:15]
        ]
        return up

    def __str__(self):
        return f'Setup(down={self.down}, up={self.up}, score={self.score})'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch 2d segmentation')
    parser.add_argument('data_path', metavar='DIR', help='path to the dataset')
    parser.add_argument('artifacts', metavar='DIR', help='path to store artifacts')

    # tournament options
    parser.add_argument('--num-contestants', type=int, default=5,
                        help='Number of contestants')
    parser.add_argument('--num-rounds', type=int, default=2,
                        help='Number of rounds')
    parser.add_argument('-s', '--early_stop', default=None, type=int,
                        help='stop early after n batches')

    # training/model options
    parser.add_argument('-nj', '--no-jit', action='store_true',
                        help='disable jit compilation for the model')
    parser.add_argument('--optimize', action='store_true',
                        help='run optimization pass in jit')
    parser.add_argument('--bloat', type=int, default=50, metavar='N',
                        help='Process N times the dataset per epoch')
    parser.add_argument('-j', '--workers', metavar='N', default=1, type=int,
                        help='number of data loader workers')
    parser.add_argument('-b', '--batch-size', metavar='N', default=1, type=int,
                        help='batch size')
    parser.add_argument('--l2', metavar='F', default=1e-5, type=float,
                        help='l2 regularization strength')
    parser.add_argument('--lr', metavar='F', default=1e-4, type=float,
                        help='learning rate (ADAM)')
    parser.add_argument('--epochs', metavar='N', default=10, type=int,
                        help='number of epochs to train for')

    args = parser.parse_args()

    with Tournament(args) as tourney:
        tourney.play()
