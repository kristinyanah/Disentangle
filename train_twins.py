import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import torchvision
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tqdm import tqdm
import math
from utils.learning_rate_scheduler import CosineAnnealingWarmUpSingle, CosineAnnealingWarmUpRestarts
import medmnist
from medmnist.dataset import DermaMNIST
from torchvision import transforms as T
import torchvision.transforms.functional as F
#from byol_pytorch import BYOL
#from byol_pytorch.byol_pytorch import RandomApply
 
from BarlowModel.barlowTwins import BarlowTwins
from BarlowModel.utils import criterions, get_byol_transforms, MultiViewDataInjector
 
from utils import LARC
import pandas as pd
import csv


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='  ')
parser.add_argument('--data', choices=['DermaMNIST'], default='DermaMNIST',
                    help='Dataset Type within MedMNIST V2 for self-supervised pretraining')
parser.add_argument('--optimizer', choices=['sgd', 'adam', 'adamw'], default='adam')
parser.add_argument('--if-LARS', action='store_true', help='whether to use LARS')
parser.add_argument('--supervised', action='store_true', help='whether to perform supervised learning')
parser.add_argument('--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=4, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=1.5e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--test', dest='test', action='store_true',
                    help='test model on test set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--scheduler', default='single', choices=[None, 'single', 'multi'], help='scheduler type. None for no scheduler.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_loss = 999999
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def main():
    args = parser.parse_args()
    #args.if_lars = True
    args.supervised = False
    args.pretrained = False
    
    args.identifier = f'data_{args.data}_lr_{args.lr}_wd_{args.weight_decay}_pretrained_{args.pretrained}_seed_{args.seed}'
    if args.supervised:
        args.fn_result = f'supervised_{args.data}'
    else:
        args.fn_result = f'selfsupervised_{args.data}'

    if not os.path.isfile(f'resnet18/{args.fn_result}.csv'):
        with open(f'resnet18/{args.fn_result}.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerow(['lr_init  ', 'wd_ratio  ', 'pretrained  ', 'seed  ', 'optimizer  ', 'loss  '])
    

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)



def main_worker(gpu, ngpus_per_node, args):
    global best_loss
    args.gpu = gpu
    args.loss_train = []
    args.loss_val = []
    args.auc_train = []
    args.auc_val = []
    args.auc_test = []
    args.acc_train = []
    args.acc_val = []
    args.acc_test = []

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, num_classes=1000)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=7)
        
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()



    

    if args.supervised:
        ds_train = DermaMNIST(split='train', download=True, root='./data/')
        ds_val = DermaMNIST(split='val', download=True, root='.data/')
        ds_test = DermaMNIST(split='test', download=True, root='./data/')
        loader_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=args.workers, pin_memory=True)
        loader_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.workers, pin_memory=True)
        loader_test = torch.utils.data.DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.workers, pin_memory=True)
    else:
        transformT, transformT1, transformEvalT = get_byol_transforms(28, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #model = BYOL(model, image_size=28, hidden_layer=-2, augment_fn=AUG_ss)

        ds_train = medmnist.dataset.__dict__[args.data](split='train', transform=MultiViewDataInjector([transformT, transformT1]), download=True, root='./data/')
        
        ds_val = medmnist.dataset.__dict__[args.data](split='val', transform=MultiViewDataInjector([transformT, transformT1]), download=True, root='./data/')

        loader_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=args.workers, pin_memory=True)
        loader_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, 
                                                    num_workers=args.workers, pin_memory=True)
       
      #  t, t1, _ = get_byol_transforms(28, [.5], [.5])
      #resnet18barlow
        model = BarlowTwins(input_size=512, output_size = 2048, backend='resnet18', pretrained_backend=True)
        #input_size=2048, output_size = 8192
        model.to(device)
        params = model.parameters()
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=1.5e-6)
        # if args.distributed:
        #     train_sampler = torch.utils.data.distributed.DistributedSampler(ds_ss_train)
        # else:
        #     train_sampler = None


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.optimizer == 'sgd':
        _optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        _optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        _optimizer = torch.optim.AdamW(model.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('Incorrect optimizer!!!')

    if bool(args.if_LARS):
        optimizer = LARC(_optimizer)
    else:
        optimizer = _optimizer

    if args.scheduler == 'multi':
        scheduler = CosineAnnealingWarmUpRestarts(_optimizer, eta_max=args.lr * math.sqrt(args.batch_size), step_total=len(loader_train) * args.epochs)

    elif args.scheduler == 'single':
        scheduler = CosineAnnealingWarmUpSingle(_optimizer, max_lr=args.lr * math.sqrt(args.batch_size),
                                                epochs=args.epochs, steps_per_epoch=len(loader_train),
                                                div_factor=math.sqrt(args.batch_size),
                                                cycle_momentum=args.momentum)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            best_auc = checkpoint['best_auc']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_auc = best_auc.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.test:
        validate(loader_test, model, criterion, args)
        return


    for epoch in range(args.epochs):
        print(f'Epoch: {epoch}')
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)

        # train for one epoch
        args.loss_train.append(train(loader_train, model, criterions, optimizer, scheduler, torch.device('cuda'), args))

        # evaluate on validation set
        args.loss_val.append(validate(loader_val, model, criterions, torch.device('cuda'), args))

        # remember best acc@1 and save checkpoint
        is_best = args.loss_val[-1] < best_loss
        best_loss = min(args.loss_val[-1], best_loss)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, args)

    df = pd.DataFrame({'lr_init':[args.lr], 'wd_ratio':[args.weight_decay], 'pretrained':[args.pretrained], 'seed':[args.seed], 'optimizer':[args.optimizer], 'loss':[best_loss]})
    df.to_csv(f'resnet18/{args.fn_result}.csv', mode='a', index=False, header=False)
    os.remove(args.identifier)





    
def train(train_loader, model, criterions, optimizer, scheduler, device, args):
    # batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1],
    #     prefix="Epoch: [{}]".format(epoch))

    # switch to train mode

    #scaler = torch.cuda.amp.GradScaler()
    model.train() 
    tk0 = tqdm(train_loader)
    epoch_loss = 0
    optimizer.zero_grad()

    for (x, x1), _ in tk0:
        #if args.gpu is not None:
        #images = images.cuda(args.gpu, non_blocking=True)
        # if torch.cuda.is_available():
        #     target = target.cuda(args.gpu, non_blocking=True)
        
       # with torch.cuda.amp.autocast():
            # compute output
        x = x.to(device)
        x1 = x1.to(device)
        fx = model(x)
        fx1 = model(x1)
        #scaler.scale(loss).backward()
        #loss.backward()
        loss = criterions(fx, fx1)
        optimizer.zero_grad()
        
        loss.backward()
       # scaler.step(optimizer)
        #model.update_moving_average()
       # scaler.update()
        optimizer.step()
        epoch_loss += loss.item()

        if args.scheduler != None:
            scheduler.step()
    epoch_loss /= float(len(train_loader))
    print(f'epoch_loss [Tr]: {epoch_loss}')
    return epoch_loss



        # # measure accuracy and record loss
        # acc1 = accuracy(output, target, topk=[1])
        # losses.update(loss.item(), images.size(0))
        # top1.update(acc1[0], images.size(0))
        #
        # # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        #
        #
        # if i % args.print_freq == 0:
        #     progress.display(i)


def validate(val_loader, model, criterions,  device, args):
    # switch to evaluate mode
    tk1 = tqdm(val_loader)
    model.eval()

    with torch.no_grad():
        epoch_loss = 0

        for (x_val, x1_val), _ in tk1:
            
            x_val = x_val.to(device)
            x1_val = x1_val.to(device)

            fx_val = model(x_val)
            fx1_val = model(x1_val)
           # if args.gpu is not None:
               # images = images.cuda(args.gpu, non_blocking=True)
            # if torch.cuda.is_available():
            #     target = target.cuda(args.gpu, non_blocking=True)

            #with torch.cuda.amp.autocast():

            loss = criterions(fx_val, fx1_val)
             
            
          
            epoch_loss += loss.item()
        epoch_loss /= float(len(val_loader))
        print(f'epoch_loss [val]: {epoch_loss}')
        return epoch_loss


def save_checkpoint(state, is_best, args):
    filename = args.identifier
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'resnet18/{args.identifier}.pth.tar')


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()