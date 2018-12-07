import matplotlib.pyplot as plt
import numpy as np
import re, os, argparse


parser = argparse.ArgumentParser()
parser.add_argument('path', nargs='+')
args = parser.parse_args()

val_re = re.compile('^Validation averagesBCE: (\d+\.\d+)$')
loss_re = re.compile('^loss: (\d+\.\d+)$')
epoch_re = re.compile('^Train epoch (\d+)$')

class Log:
    def __init__(self, name):
        self.name = name
        self.vals = []
        self.losses = []
        self.epochs = []

    def update(self, val, loss, epoch):
        self.vals.append(val)
        self.losses.append(loss)
        self.epochs.append(epoch)

    def get(self):
        vals = np.array(self.vals)
        losses = np.array(self.losses)
        epochs = np.array(self.epochs)

        return vals, losses, epochs

def parse_file(fname):
    log = Log(fname)
    losses = []
    for line in open(fname, 'r'):
        m_epoch = epoch_re.match(line)
        if m_epoch:
            epoch = int(m_epoch.group(1))

        m_val = val_re.match(line)
        if m_val:
            mean_loss = np.array(losses).mean()
            val = float(m_val.group(1))
            log.update(val, mean_loss, epoch)
            losses = []

        m_loss = loss_re.match(line)
        if m_loss:
            losses.append(float(m_loss.group(1)))

    return log

log_re = re.compile('^log(\d+).log$')

def parse_files(search_paths=['.']):
    parsed = dict()

    for base in search_paths:
        for root, dirs, files in os.walk(base):
            for file in files:
                m = log_re.match(file)
                if not m:
                    continue

                path = os.path.join(root, file)
                try:
                    log = parse_file(path)
                except Exception as e:
                    print(f'Exception {e} happened when processing {path}')
                    continue

                if len(log.epochs) > 1:
                    parsed[path] = log

    return parsed

parsed = parse_files(args.path)

fig, (train_ax, test_ax) = plt.subplots(2, 1)
train_ax.set_title('training')
test_ax.set_title('testing')

for i, path in enumerate(parsed):
    color = f'C{i % 10}'

    log = parsed[path]
    train_ax.plot(log.epochs, log.losses, color, label=path)
    test_ax.plot(log.epochs, log.vals, color)

handles, labels = train_ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

plt.show()
