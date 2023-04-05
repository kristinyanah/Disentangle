import argparse 
import torch
import torchvision
import utils
import simclr
from PIL import Image
import os
import numpy as np
import medmnist
from medmnist import INFO, Evaluator, RetinaMNIST, DermaMNIST, BloodMNIST
import pandas as pd
import shutil, time, os, requests, random, copy
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, datasets
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as F



# making a command line interface
parser = argparse.ArgumentParser(description=" ")

parser.add_argument('datapath', type=str ,help="Path to the data root folder which contains train and test folders")

parser.add_argument('respath', type=str, help="Path to the results directory where the saved model and evaluation graphs would be stored. ")

parser.add_argument('-bs','--batch_size',default=250, type=int, help="The batch size for self-supervised training")

parser.add_argument('-nw','--num_workers',default=2,type=int,help="The number of workers for loading data")

parser.add_argument('-c','--cuda',action='store_true')

parser.add_argument('--multiple_gpus', action='store_true')

class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, args):
        self.args = args
        
        with open(os.path.join(args.datapath, "train","names.txt")) as f:
            self.filenames = f.read().split('\n')
 
    def __len__(self):
        return len(self.filenames)

    def tensorify(self, img):
        return torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
            torchvision.transforms.ToTensor()(img)
            )

    def augmented_image(self, img):
        return utils.transforms.get_color_distortion(1)(
            torchvision.transforms.RandomResizedCrop(224)(img)
            )    

    def __getitem__(self, idx):
        img = torchvision.transforms.Resize((224, 224))(
                                Image.open(os.path.join(args.datapath, 'train', self.filenames[idx])).convert('RGB')
                            )
        return {
        'image1':self.tensorify(
            self.augmented_image(img)
            ), 
        'image2': self.tensorify(
            self.augmented_image(img)
            )
        }

class ContrastiveLearningViewGenerator(object):
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views
class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        #color_jitter = transforms.ColorJitter(0.9 * s, 0.9 * s, 0.9 * s, 0.2 * s)
        data_transforms = transforms.Compose([#transforms.RandomResizedCrop(size=size, scale=(0.8, 1.0)),
                                              #transforms.RandomHorizontalFlip(),
                                              #transforms.RandomRotation(degrees=180, interpolation=F.InterpolationMode.BILINEAR),
                                              #transforms.RandomApply([color_jitter], p=0.8),
                                              #transforms.RandomGrayscale(p=0.2),
                                              #GaussianBlur(kernel_size=int(0.2 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'retina': lambda: RetinaMNIST(split='train',  download=True),
                        'bloodmnist': lambda: BloodMNIST(split='train',  download=True),
                        'dermamnist': lambda: DermaMNIST(split='train',  download=True),}
        dataset_fn = valid_datasets[name]
        return dataset_fn()

class C10DataGen(Dataset):
    def __init__(self,phase,imgarr,s = 0.5):
        self.phase = phase
        self.imgarr = imgarr
        self.s = s
        self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                              
                                             # transforms.RandomApply([T.GaussianBlur((3, 3), (1.0, 2.0))],p=0.2),
                                              transforms.RandomRotation(degrees=180, interpolation=F.InterpolationMode.BILINEAR),
                                              transforms.RandomResizedCrop(28,(0.8,1.0)),
                            
                                              transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.9*self.s, 
                                                                                                                 0.9*self.s, 
                                                                                                                 0.9*self.s, 
                                                                                                                 0.2*self.s)], p = 0.3),
                                                                  #transforms.RandomGrayscale(p=0.2)
                                                                 ])])

    def __len__(self):
        return self.imgarr.shape[0]

    def __getitem__(self,idx):
        
        x = self.imgarr[idx] 
        #print(x.shape)
        x = x.astype(np.float32)/255.0

        x1 = self.augment(torch.from_numpy(x))
        x2 = self.augment(torch.from_numpy(x))
        
        x1 = self.preprocess(x1)
        x2 = self.preprocess(x2)
        
        return x1, x2

    #shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        self.imgarr = self.imgarr[random.sample(population = list(range(self.__len__())),k = self.__len__())]

    def preprocess(self,frame):
        MEAN = np.mean(self.imgarr/255.0,axis=(0,2,3),keepdims=True)
        STD = np.std(self.imgarr/255.0,axis=(0,2,3),keepdims=True)
        frame = (frame-MEAN)/STD
        return frame
    
    #applies randomly selected augmentations to each clip (same for each frame in the clip)
    def augment(self, frame, transformations = None):
        
        if self.phase == 'train':
            frame = self.transforms(frame)
        else:
            return frame
        
        return frame
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    model = utils.model.get_model(args)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.001, 
        weight_decay=1e-4
        )
    
    
    dataset = ContrastiveLearningDataset('bloodmnist')
    train_dataset = dataset.get_dataset('bloodmnist', 2)
    images = train_dataset.imgs
    labels = train_dataset.labels
    images = images.reshape((-1,3,28,28))
    images = images.astype(np.float)
    labels = labels.astype(np.int)
    
    MEAN = np.mean(images/255.0,axis=(0,2,3),keepdims=True)
    STD = np.std(images/255.0,axis=(0,2,3),keepdims=True)
    dataloaders = torch.utils.data.DataLoader(
        C10DataGen('train', images), 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
        )
    
    loss_fn = utils.ntxent.loss_function
    simclrobj = simclr.SimCLR(model, optimizer, dataloaders, loss_fn)
    simclrobj.train(args, 200, 10)
