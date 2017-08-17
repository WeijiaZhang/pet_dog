# -*- coding: utf-8 -*-
from __future__ import print_function

from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torchvision
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os, sys, h5py, gc, argparse, codecs, shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument('--ffpath', required=True, help='path for feature file')
parser.add_argument('--model', required=True, help='cnn model')
parser.add_argument('--crop', required=False, action='store_true', help='dog detection')
opt = parser.parse_args()
print(opt)

num_classes = 134
device_id = 2
device_id_all = [2]
train_val_path = '../../dataset/train/train_val_data.txt'

#train = pd.read_csv('../../dataset/train/data_train_image.txt', header=None, sep=' ', names=['img', 'label', 'url'])
#val = pd.read_csv('../../dataset/train/val.txt', header=None, sep=' ', names=['img', 'label', 'url'])

# lbl = LabelEncoder()
# train['label'] = lbl.fit_transform(train['label'].values)

#train['img'] = '../../dataset/train/train_image/' + train['img'] + '.jpg'
#val['img'] = '../../dataset/train/val_image/' + val['img'] + '.jpg'

train_val = pd.read_csv(train_val_path, header=None, sep='\t', names=['img', 'label'])

train_val['img'] = '../' + train_val['img']

state_name_cv_3 = {'densenet161': '161_cv10_256_1/densenet161_cv_3_1879', 
	      'densenet169': '169_cv10_256_1/densenet169_cv_3_1948',
	      'resnet101': '101_cv10_256_1/resnet101_cv_3_1902',
	      'resnet152': '152_cv10_256_1/resnet152_cv_3_1949',
	      'resnet50': '50_cv10_256_1/resnet50_cv_3_196',
	      'inception_v3': 'v3_cv10_299_1/inception_v3_cv_3_1947',
             }
	
state_name_cv_5 = {'densenet161': '161_cv10_256_1/densenet161_cv_5_191', 
	      'densenet169': '169_cv10_256_1/densenet169_cv_5_1909',
	      'resnet101': '101_cv10_256_1/resnet101_cv_5_1913',
	      'resnet152': '152_cv10_256_1/resnet152_cv_5_1931',
	      'resnet50': '50_cv10_256_1/resnet50_cv_5_1948',
	      'inception_v3': 'v3_cv10_299_1/inception_v3_cv_5_192',
             }
	
state_name_cv_7 = {'densenet161': '161_cv10_256_1/densenet161_cv_7_1882', 
	      'densenet169': '169_cv10_256_1/densenet169_cv_7_79',
	      'resnet101': '101_cv10_256_1/resnet101_cv_7_1886',
	      'resnet152': '152_cv10_256_1/resnet152_cv_7_192',
	      'resnet50': '50_cv10_256_1/resnet50_cv_7_1967',
	      'inception_v3': 'v3_cv10_299_1/inception_v3_cv_7_1944',
             }
		
state_name_cv_9 = {'densenet161': 'ensemble_1/densenet161_1892', 
	      'densenet169': 'ensemble_1/densenet169_1926',
	      'resnet101': 'ensemble_1/resnet101_1924',
	      'resnet152': 'ensemble_1/resnet152_1918',
	      'resnet50': 'ensemble_1/resnet50_1946',	
	      'inception_v3': 'ensemble_1/inception_v3_1981',
             }
	     
state_name = state_name_cv_7
state_path = {}
for s_name in state_name.keys():
    state_path[s_name] = '../../results/best_params/' + state_name[s_name] + '_ckpt.t7'


# 删除标签不一致的情况
#train_ = train[~train['img'].duplicated(keep=False)]
#train_val = pd.concat([train, val], axis=0, ignore_index=True)

# dog_crop = h5py.File('./yolo_kuhuang.h5', 'r')
# dog_crop_img = dog_crop.keys()


# 读取目标img文件，归一化到指定大小
def read_img(img_file, size=(224, 224), logging=False):
    imgs = []
    for img_path in img_file:
        img = Image.open(img_path)

        # if opt.crop and (img_path.split('/')[-1] in dog_crop_img):
        #     img = img.crop(dog_crop[img_path.split('/')[-1]][:])
            # print(img_path)
        imgs.append(img)
    return imgs

network = opt.model
if network == 'resnet18':
    model_conv = torchvision.models.resnet18(pretrained=True)
    model_conv = nn.Sequential(*list(model_conv.children())[:-1])
    featurenum = 512
    batchsize = 64
elif network == 'resnet34':
    model_conv = torchvision.models.resnet34(pretrained=True)
    model_conv = nn.Sequential(*list(model_conv.children())[:-1])
    featurenum = 512
    batchsize = 64
elif network == 'resnet50':
    model_conv = torchvision.models.resnet50()
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_classes)
    model_conv = nn.DataParallel(model_conv, device_id_all)
    
    state = torch.load(state_path[network])
    print('load state dict from %s' % state_path[network])
    model_conv.load_state_dict(state['model_param'])
    
    model_conv.module.fc = nn.Sequential(*list(model_conv.module.fc.children())[:-1])
    featurenum = 2048
    batchsize = 64
elif network == 'resnet101':
    model_conv = torchvision.models.resnet101()
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_classes)
    model_conv = nn.DataParallel(model_conv, device_id_all)
    
    state = torch.load(state_path[network])
    print('load state dict from %s' % state_path[network])
    model_conv.load_state_dict(state['model_param']) 

    model_conv.module.fc = nn.Sequential(*list(model_conv.module.fc.children())[:-1])
    featurenum = 2048
    batchsize = 48
elif network == 'resnet152':
    model_conv = torchvision.models.resnet152()
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_classes)
    model_conv = nn.DataParallel(model_conv, device_id_all)
    
    state = torch.load(state_path[network])
    print('load state dict from %s' % state_path[network])
    model_conv.load_state_dict(state['model_param'])

    model_conv.module.fc = nn.Sequential(*list(model_conv.module.fc.children())[:-1])
    featurenum = 2048
    batchsize = 48
elif network == 'vgg11_bn':
    model_conv = torchvision.models.vgg11_bn(pretrained=True)
    model_conv.classifier = nn.Sequential(*list(model_conv.classifier.children())[:-1])
    featurenum = 4096
    batchsize = 64
elif network == 'vgg13_bn':
    model_conv = torchvision.models.vgg13_bn(pretrained=True)
    model_conv.classifier = nn.Sequential(*list(model_conv.classifier.children())[:-1])
    featurenum = 4096
    batchsize = 64
elif network == 'vgg16_bn':
    model_conv = torchvision.models.vgg16_bn(pretrained=True)
    model_conv.classifier = nn.Sequential(*list(model_conv.classifier.children())[:-1])
    featurenum = 4096
    batchsize = 64
elif network == 'vgg19_bn':
    model_conv = torchvision.models.vgg19_bn(pretrained=True)
    model_conv.classifier = nn.Sequential(*list(model_conv.classifier.children())[:-1])
    featurenum = 4096
    batchsize = 60
elif network == 'densenet121':
    model_conv = torchvision.models.densenet121()
    num_ftrs = model_conv.classifier.in_features
    model_conv.classifier = nn.Linear(num_ftrs, num_classes)
    model_conv = nn.DataParallel(model_conv, device_id_all)
    
    state = torch.load(state_path[network])
    print('load state dict from %s' % state_path[network])
    model_conv.load_state_dict(state['model_param'])
 
    model_conv.module.classifier = nn.Sequential(*list(model_conv.module.classifier.children())[:-1])
    featurenum = 1024
    batchsize = 64
elif network == 'densenet161':
    model_conv = torchvision.models.densenet161()
    num_ftrs = model_conv.classifier.in_features
    model_conv.classifier = nn.Linear(num_ftrs, num_classes)
    model_conv = nn.DataParallel(model_conv, device_id_all)
    
    state = torch.load(state_path[network])
    print('load state dict from %s' % state_path[network])
    model_conv.load_state_dict(state['model_param'])
 
    model_conv.module.classifier = nn.Sequential(*list(model_conv.module.classifier.children())[:-1])
    featurenum = 2208
    batchsize = 32
elif network == 'densenet169':
    model_conv = torchvision.models.densenet169()
    num_ftrs = model_conv.classifier.in_features
    model_conv.classifier = nn.Linear(num_ftrs, num_classes)
    model_conv = nn.DataParallel(model_conv, device_id_all)
    
    state = torch.load(state_path[network])
    print('load state dict from %s' % state_path[network])
    model_conv.load_state_dict(state['model_param'])
 
    model_conv.module.classifier = nn.Sequential(*list(model_conv.module.classifier.children())[:-1])
    featurenum = 1664
    batchsize = 48
elif network == 'densenet201':
    model_conv = torchvision.models.densenet201()
    num_ftrs = model_conv.classifier.in_features
    model_conv.classifer = nn.Linear(num_ftrs, num_classes)
    model_conv = nn.DataParallel(model_conv, device_id_all)
    
    state = torch.load(state_path[network])
    print('load state dict from %s' % state_path[network])
    model_conv.load_state_dict(state['model_param'])
 
    model_conv.module.classifier = nn.Sequential(*list(model_conv.module.classifier.children())[:-1])
    featurenum = 1920
    batchsize = 32
elif network == 'inception_v3':
    model_conv = torchvision.models.inception_v3(pretrained=True)
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_classes)
    model_conv = nn.DataParallel(model_conv, device_id_all)
    
    # state = torch.load(state_path[network])
    # print('load state dict from %s' % state_path[network])
    # model_conv.load_state_dict(state['model_param'])
 
    model_conv.module.fc = nn.Sequential(*list(model_conv.module.fc.children())[:-1])
    featurenum = 2048
    batchsize = 64

#device_id_all = [0, 1, 2, 3]
#model_conv = nn.DataParallel(model_conv, device_ids=device_id_all)
if len(device_id_all) > 1:
    model_conv = model_conv.cuda()

model_conv.eval()
print(network, featurenum)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
# Inception 输入大小是299
if 'inception' in network:
    tr = {
        'train': transforms.Compose([
            # transforms.Scale(320),
            # transforms.CenterCrop(299),
            transforms.RandomSizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]),
        'val': transforms.Compose([
            transforms.Scale(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            normalize])
    }
elif 'vgg' in network:
    tr = {
        'train': transforms.Compose([
            # transforms.Scale(320),
            # transforms.CenterCrop(299),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
    }
else:
    tr = {
        'train': transforms.Compose([
            # transforms.Scale(278),
            # transforms.CenterCrop(256),
            transforms.RandomSizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]),
        'val': transforms.Compose([
            transforms.Scale(278),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            normalize])
    }

# train_val = train_val.iloc[: 100]
train_feature = []
for idx in range(0, train_val.shape[0], batchsize):
    if idx + batchsize < train_val.shape[0]:
        ff = read_img(train_val['img'].iloc[idx: idx + batchsize].values)
        ff = [tr['val'](x) for x in ff]
        ff = torch.stack(ff)
	ff = model_conv(Variable(ff.cuda(device_id)))
	ff = ff.view(-1, featurenum)
        train_feature.append(ff.data.cpu().numpy())
        del ff
        gc.collect()
    else:
        ff = read_img(train_val['img'].iloc[idx:].values)
        ff = [tr['val'](x) for x in ff]
        ff = torch.stack(ff)
        ff = model_conv(Variable(ff.cuda(device_id))).view(-1, featurenum)
        train_feature.append(ff.data.cpu().numpy())
        del ff
        gc.collect()
    if (idx // batchsize) % 20  == 19:
	print('Train', idx, train_val.shape[0])
train_feature = np.array(train_feature)

#test = os.listdir('../../dataset/test/test_image/')
#test = ['../../dataset/test/test_image/' + x for x in test]

test_path = '../../dataset/test/test_info.txt'
test_df = pd.read_csv(test_path, header=None, sep='\t', names=['img', 'label'])
test = '../../dataset/test/test_image/' + test_df['img'].values + '.jpg'

# test = test[: 100]
test_feature = []
for idx in range(0, len(test), batchsize):
    if idx + batchsize < len(test):
        ff = read_img(test[idx: idx + batchsize])
        ff = [tr['val'](x) for x in ff]
        ff = torch.stack(ff)
        ff = model_conv(Variable(ff.cuda(device_id))).view(-1, featurenum)
        test_feature.append(ff.data.cpu().numpy())
        del ff
        gc.collect()
    else:
        ff = read_img(test[idx:])
        ff = [tr['val'](x) for x in ff]
        ff = torch.stack(ff)
        ff = model_conv(Variable(ff.cuda(device_id))).view(-1, featurenum)
        test_feature.append(ff.data.cpu().numpy())
        del ff
        gc.collect()
    if (idx // batchsize) % 20 == 19:
	print('Test', idx, len(test))
test_feature = np.array(test_feature)

train_feature = np.concatenate(train_feature, 0).reshape(-1, featurenum)
test_feature = np.concatenate(test_feature, 0).reshape(-1, featurenum)

print('write train and test features to file %s...' % opt.ffpath)
with h5py.File(opt.ffpath, "w") as f:
    f.create_dataset("train_feature", data=train_feature)
    f.create_dataset("test_feature", data=test_feature)
