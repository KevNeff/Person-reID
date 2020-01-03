from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from model import ft_net, ft_net_dense, PCB, PCB_test
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
#parser.add_argument('--PCB', action='store_true', help='use PCB' )

opt = parser.parse_args()
gpu_ids = []
str_ids = opt.gpu_ids.split(',')
for str_id in str_ids:
	id = int(str_id)	
	if id >=0:
		gpu_ids.append(id)
#which_epoch = opt.which_epoch
name = opt.name
#test_dir = opt.test_dir

data_dir = "/scratch/user/anuragdiisc.ac.in/Dataset/valSet"
query_dir_path = "/scratch/user/anuragdiisc.ac.in/Dataset/valSet/query"
gallery_dir_path = "/scratch/user/anuragdiisc.ac.in/Dataset/valSet/gallery"
model = "ft_ResNet50"
log_dir = "/scratch/user/anuragdiisc.ac.in/Dataset/valSet/log"

#gpu_ids = []
#for str_id in str_ids:
#    id = int(str_id)
#    if id >=0:
#        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((288,144), interpolation=3),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

data_transforms = transforms.Compose( transform_train_list )

#model_structure = ft_net(751)
#model = load_network(model_structure)
#model = model.eval()
#use_gpu = torch.cuda.is_available()
#if use_gpu:
#        model = model.cuda()

def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


model_structure = ft_net(751)
model = load_network(model_structure)
model = model.eval()
use_gpu = torch.cuda.is_available()

###changed here
#if use_gpu:
#    model = model.cuda()


class Dataset(Dataset):
    def __init__(self, path, transform):
        self.dir = path
        self.image = [f for f in os.listdir(self.dir) if f.endswith('png')]
        self.transform = transform

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        name = self.image[idx]

        img = Image.open(os.path.join(self.dir, name))
        img = self.transform(img)

        return {'name': name.replace('.png', ''), 'img': img}

def extractor(model, dataloader):
    def fliplr(img):
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    test_names = []
    test_features = torch.FloatTensor()

    for batch, sample in enumerate(dataloader):
        names, images = sample['name'], sample['img']
        print("------------------",images.size()) 
        #ff = model(Variable(images.cuda(), volatile=True))[0].data.cpu()
        #ff = model(Variable(images, volatile=True))[0].data.cpu()
        ff = model(Variable(images, volatile=True)).data.cpu()
        #ff = ff + model(Variable(fliplr(images).cuda(), volatile=True))[0].data.cpu()
        #ff = ff + model(Variable(fliplr(images), volatile=True))[0].data.cpu()
        ff = ff + model(Variable(fliplr(images), volatile=True)).data.cpu()
        print(ff.shape,"*************************")
        ff_norm = torch.norm(ff, p = 2, dim = 1, keepdim = True)
        print(ff_norm.shape,"====================")
        ff = ff.div(torch.norm(ff, p=2, dim=1, keepdim=True).expand_as(ff))

        test_names = test_names + names
        test_features = torch.cat((test_features, ff), 0)

    return test_names, test_features

#def load_network(network):
#    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
#    network.load_state_dict(torch.load(save_path))
#    return network

#model_structure = ft_net(751)
#model = load_network(model_structure)
#model = model.eval()
#use_gpu = torch.cuda.is_available()
#if use_gpu:
#	model = model.cuda()


image_datasets = {'val':{'gallery': Dataset(gallery_dir_path, data_transforms),'query': Dataset(query_dir_path, data_transforms)}}
dataloaders = {'val':{x: torch.utils.data.DataLoader(image_datasets['val'][x], batch_size=opt.batchsize,shuffle=False, num_workers=16) for x in ['gallery','query']}}


for dataset in ['val']:
    for subset in ['query', 'gallery']:
        test_names, test_features = extractor(model, dataloaders[dataset][subset])
        results = {'names': test_names, 'features': test_features.numpy()}
        scipy.io.savemat(os.path.join(log_dir, 'feature_%s_%s.mat' % (dataset, subset)), results)



