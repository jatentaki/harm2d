import torch, os
from utils import AvgMeter, open_file, print_dict, size_adaptive_, fmt_value
from tqdm import tqdm

def load_checkpoint(path):
    cp = torch.load(path)
    return cp['epoch'], cp['score'], cp['model'], cp['optim']

def save_checkpoint(epoch, score, model, optim, path=None, fname=None):
    if path is None:
        raise ValueError("No path specified for save_checkpoint")

    cp = {
        'epoch': epoch,
        'score': score,
        'model': model.state_dict(),
        'optim': optim.state_dict()
    }

    if not os.path.exists(path):
        os.makedirs(path)
        print("Created save directory {}".format(path))
    else:
        if not os.path.isdir(path):
            fmt = "{} is not a directory"
            msg = fmt.format(path)
            raise IOError(msg)

    fname = fname if fname else 'epoch{}.pth.tar'.format(epoch)
    torch.save(cp, path + os.path.sep + fname)

def inspect(network, loader, path, early_stop=None, criteria=[]):
    network.eval()
    path += os.path.sep       
    maybe_make_dir(path)
    
    with tqdm(total=len(loader), dynamic_ncols=True) as progress, torch.no_grad():
        for i, args in enumerate(loader):
            if i == early_stop:
                break

            if torch.cuda.is_available():
                input = args[0].cuda()      

            directory = path + os.path.sep + str(i) 
    
            # prediction
            prediction = network(input)
            heatmap = torch.sigmoid(prediction)
            heatmap = heatmap.cpu()[0, 0].numpy()
            hm_path = directory + os.path.sep + 'heatmap.npy'                     
            maybe_make_dir(hm_path, silent=True)          
            np.save(hm_path, heatmap)
                      
            # image 
            image = input.cpu()[0].numpy().transpose(1, 2, 0)
            np.save(directory + os.path.sep + 'image.npy', image)

            # ground truth             
            g_truth = args[-1][0, 0].numpy()
            np.save(directory + os.path.sep + 'g_truth.npy', g_truth)

            progress.update(1) 


def train(network, dataset, loss_fn, optimizer, epoch, early_stop=None, logger=None):
    def optimization_step(*args):
        if torch.cuda.is_available():
            args = [a.cuda() for a in args]

        prediction = network(args[0])
        optimizer.zero_grad()
        loss = loss_fn(prediction, *args[1:])
        loss.backward()
        optimizer.step()

        return loss

    loss_meter = AvgMeter()
    network.train()
    msg = 'Train epoch {}'.format(epoch)
    print(msg)
    if logger is not None:
        logger.add_msg(msg)

    with tqdm(total=len(dataset), dynamic_ncols=True) as progress:
        for i, args in enumerate(dataset):
            if i == early_stop:
                break

            loss = optimization_step(*args)

            loss_meter.update(loss.item())

            progress.update(1)
            progress.set_postfix(loss=loss_meter.last, mean=loss_meter.avg)

            if logger is not None:
                logger.add_dict({'loss': loss_meter.last})

    return loss_meter.avg


def test(network, dataset, criteria, early_stop=None, logger=None, callbacks=[]):
    if criteria == []:
        raise ValueError("Empty criterion list")

    network.eval()
    meters = [AvgMeter() for c in criteria]

    if logger is not None:
        logger.add_msg('Validation')

    with tqdm(total=len(dataset), dynamic_ncols=True) as progress, torch.no_grad():
        for i, args in enumerate(dataset):
            if i == early_stop:
                break

            if torch.cuda.is_available():
                args = [a.cuda() for a in args]

            prediction = network(args[0])

            for callback in callbacks:
                callback(prediction, *args[1:])

            for criterion, meter in zip(criteria, meters):
                perf = criterion(prediction, *args[1:])
                if isinstance(perf, torch.Tensor):
                    meter.update(perf.item())
                else:
                    meter.update(perf)

            progress.update(1)

            criteria_dict = {
                c.name: m.last for c, m in zip(criteria, meters)
            }

            if logger is not None:
                logger.add_dict(criteria_dict)

    means = print_dict({c.name: m.avg for c, m in zip(criteria, meters)})
    print('Validation averages\n\t' + means)
    if logger is not None:
        logger.add_msg('Validation averages' + means)
    return meters[0].avg


class Logger:
    def __init__(self, path):
        self.path = path
        self.file = None


    def __enter__(self):
        self.open()
        return self


    def __exit__(self, *errs):
        self.close()
    

    def open(self):
        # find a unique log name
        i = 0
        while True:
            proposed_name = self.path + str(i) + '.log'
            if os.path.isfile(proposed_name):
                print('Found previous log', proposed_name)
                i += 1
            else:
                break
            
        self.path = proposed_name
        self.file = open_file(self.path, 'w', buffering=512)
        print('Logging to', self.path)


    def close(self):
        self.file.close()
        self.file = None


    def add_msg(self, msg):
        self.file.write(msg + '\n')


    def add_dict(self, dict_):
        self.add_msg(print_dict(dict_, prec=6))
