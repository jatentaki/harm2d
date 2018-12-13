import torch, os
import numpy as np
from tqdm import tqdm
from torchvision.utils import make_grid

from utils import AvgMeter, open_file, print_dict, maybe_make_dir

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

def inspect(network, loader, path, early_stop=None):
    network.eval()
    path += os.path.sep       
    maybe_make_dir(path)
    
    progress = tqdm(total=len(loader), dynamic_ncols=True)
    with progress, torch.no_grad():
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


def train(network, dataset, loss_fn, optimizer, epoch, writer,
          early_stop=None, batch_multiplier=1):

    def optimization_step(i, *args, save=False):
        if i % batch_multiplier == 0:
            optimizer.zero_grad()

        if torch.cuda.is_available():
            args = [a.cuda() for a in args]

        prediction = network(args[0])
        loss = loss_fn(prediction, *args[1:])
        loss.backward()

        if (i+1) % batch_multiplier == 0:
            optimizer.step()

        if save:
            pred = torch.sigmoid(prediction)
            writer.add_image('Train/prediction', make_grid(pred), epoch)
            writer.add_image('Train/image', make_grid(args[0]), epoch)

        return loss

    loss_meter = AvgMeter()
    network.train()

    progress = tqdm(total=len(dataset), dynamic_ncols=True)
    with progress:
        for i, args in enumerate(dataset):
            if i == early_stop:
                break

            loss = optimization_step(i, *args, save=i == 0)

            writer.add_scalar('Train/loss', loss.item(), epoch)
            progress.update(1)
            loss_meter.update(loss.item())
            progress.set_postfix(loss=loss_meter.last, mean=loss_meter.avg)

    writer.add_scalar('Train/loss_mean', loss_meter.avg, epoch)
    return loss_meter.avg


def test(network, dataset, loss_fn, criteria, epoch, writer, early_stop=None):
    network.eval()
    loss_meter = AvgMeter()

    progress = tqdm(total=len(dataset), dynamic_ncols=True)
    with progress, torch.no_grad():
        for i, args in enumerate(dataset):
            if i == early_stop:
                break

            if torch.cuda.is_available():
                args = [a.cuda() for a in args]

            prediction = network(args[0])
            loss = loss_fn(prediction, *args[1:]).item()

            loss_meter.update(loss)

            writer.add_scalar('Test/loss', loss, epoch)
            for criterion in criteria:
                value = criterion(prediction, *args[1:])
                writer.add_scalar(f'Test/{criterion.name}', value.item(), epoch)
            progress.update(1)

    pred = torch.sigmoid(prediction)
    writer.add_image('Test/prediction', make_grid(pred), epoch)
    writer.add_image('Test/image', make_grid(args[0]), epoch)

#    grid = make_grid(torch.sigmoid(prediction))
#    writer.add_image('Test/image', grid, epoch)

    writer.add_scalar('Test/loss_mean', loss_meter.avg, epoch)
