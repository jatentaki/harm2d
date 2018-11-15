import re, os
import numpy as np
import matplotlib.pyplot as plt

setup_re = re.compile('^Setup (.+?)$')
epoch_re = re.compile('^Train epoch (\d+)$')
loss_re = re.compile('^loss: (\d*\.\d*)$')
bce_re = re.compile('^Validation averagesBCE: (\d*\.\d*)$')

def parse_log(logp):
    with open(logp, 'r') as logf:
        lines = logf.readlines()

    epochs = []
    losses = []
    setup = None
    for line in lines:
        m_setup = setup_re.match(line)
        if m_setup:
            setup = m_setup.group(1)

#        m_epoch = epoch_re.match(line)
#        if m_epoch:
#            if losses != []:
#                mean = np.array(losses).mean()
#
#            pass

        m_loss = loss_re.match(line)
        if m_loss:
            losses.append(float(m_loss.group(1)))

        m_bce = bce_re.match(line)
        if m_bce:
            train_loss = np.array(losses).mean()
            test_loss = float(m_bce.group(1))
            
            epochs.append((train_loss, test_loss))
            losses = []

    if setup is None:
        return None
    else:
        return setup, np.array(epochs).T

class Experiment:
    def __init__(self, setup, tr1, te1, tr2, te2):
        self.setup = setup
        self.tr1 = tr1
        self.te1 = te1
        self.tr2 = tr2
        self.te2 = te2

def enumerate_logs():
    labels = []
    for exp in range(5):
        for rep in range(1, 3):
            path_fmt = 'artifacts_harmonic_{}/rep_{}/log0.log'
            log_path = path_fmt.format(exp, rep)
            if not os.path.isfile(log_path):
                continue

            parsed = parse_log(log_path)
            if parsed is not None:
                setup, (train, test) = parsed
                test += np.random.randn(*test.shape) * 0.00005
                plt.plot(train, 'C{}--'.format(exp))
                if rep == 1:
                    label, = plt.plot(test, 'C{}-'.format(exp), label=setup)
                    labels.append(label)
                else:
                    plt.plot(test, 'C{}-'.format(exp))
    plt.legend(handles=labels)

enumerate_logs()
plt.show()
