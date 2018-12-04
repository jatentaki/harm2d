import matplotlib.pyplot as plt
import numpy as np
import re, os, argparse


parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()

val_re = re.compile('^Validation averagesBCE: (\d+\.\d+)$')
loss_re = re.compile('^loss: (\d+\.\d+)$')

def parse_file(fname):
    validations = []
    current_losses = []
    losses = []
    for line in open(fname, 'r'):
        m_val = val_re.match(line)
        if m_val:
            validations.append(float(m_val.group(1)))
            losses.append(np.array(current_losses).mean())
            current_losses = []

        m_loss = loss_re.match(line)
        if m_loss:
            current_losses.append(float(m_loss.group(1)))

    return np.array(validations), np.array(losses)


def parse_files(base='.'):
    file_re = re.compile('^artifacts_(\d+)_(harmonic|baseline)_(no_rot|rotate)$')

    parsed = dict()
    for dir in os.listdir(base):
        m = file_re.match(dir)
        if not m:
            continue

        logp = os.path.join(base, dir, 'log0.log')

        num = int(m.group(1))
        type = m.group(2)
        aug = m.group(3) == 'rotate'

        val, train = parse_file(logp)

        if num in parsed:
            parsed[num][(type, aug)] = (val, train)
        else:
            parsed[num] = {(type, aug): (val, train)}

    return parsed

parsed = parse_files(args.path)
styles = {
    ('harmonic', True): '-',
    ('harmonic', False): 'o',
    ('baseline', True): '--',
    ('baseline', False): 'x',
}

fig, (train_ax, test_ax) = plt.subplots(2, 1)
train_ax.set_title('training')
test_ax.set_title('testing')

for i, num in enumerate(parsed):
    color = f'C{i}'

    for type_aug in parsed[num]:
        type, aug = type_aug
        val, train = parsed[num][type_aug]
        style = styles[type_aug]

        if type != 'harmonic':
            continue
        label = f'{num}, {type}, {aug}'
        train_ax.plot(train[:25], color + style, label=label)
        test_ax.plot(val[:25], color + style)

handles, labels = train_ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

plt.show()
