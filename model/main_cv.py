from __future__ import print_function
import torch
import torchvision

import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

from sklearn.model_selection import KFold
from train_val_cv import *
from data_preprocess.img_process import *

train_data_path = '../processed/train_val10_299_1.t7'

use_cuda = torch.cuda.is_available()
device_id = 2
device_id_part1 = [0, 1]
device_id_part2 = [2, 3]
device_id_all = [0, 1, 2, 3]

# model_res101 = models.resnet101(pretrained=True)
# num_ftrs = model_res101.fc.in_features
# model_res101.fc = nn.Linear(num_ftrs, NUM_CLASSES)
# model_res101 = nn.DataParallel(model_res101, device_ids=device_id_all)


def get_resnet_152(num_classes, device_ids):
    model_152 = models.resnet152(pretrained=True)
    num_ftrs = model_152.fc.in_features
    model_152.fc = nn.Linear(num_ftrs, num_classes)
    model_152 = nn.DataParallel(model_152, device_ids=device_ids)
    return model_152


def get_densenet_161(num_classes, device_ids):
    model_161 = models.densenet161(pretrained=True)
    num_ftrs = model_161.classifier.in_features
    model_161.classifier = nn.Linear(num_ftrs, num_classes)
    model_161 = nn.DataParallel(model_161, device_ids=device_ids)
    return model_161

model_cv = []
for i in range(NUM_CV):
    model_cv.append(get_densenet_161(NUM_CLASSES, device_id_all))

# model_inc_v3 = models.inception_v3(pretrained=True)
# num_ftrs = model_inc_v3.fc.in_features
# model_inc_v3.fc = nn.Linear(num_ftrs, NUM_CLASSES)
# model_inc_v3 = nn.DataParallel(model_inc_v3, device_ids=device_id_all)

if use_cuda:
    # model_inc_v3.cuda()
    # model_res101.cuda()
    for model_i in model_cv:
        model_i.cuda()

cross_entropy = nn.CrossEntropyLoss()
# sgd_res101 = optim.SGD(model_res101.parameters(), lr=3e-4, momentum=0.9, weight_decay=5e-3)
# sgd_inc_v3 = optim.SGD(model_inc_v3.parameters(), lr=3e-4, momentum=0.9, weight_decay=5e-3)

sgd_cv = []
for model_i in model_cv:
    sgd_i = optim.SGD(model_i.parameters(), lr=3e-4, momentum=0.9,
                      weight_decay=5e-3)
    sgd_cv.append(sgd_i)


# read train dataset
imgs_train, imgs_name_train, classes_train = [], [], []
if os.path.isfile(train_data_path):
    train_data = torch.load(train_data_path)
    imgs_train, imgs_name_train, classes_train = train_data['imgs'], train_data['imgs_name'], train_data['classes']
    print('load train data from %s' % train_data_path)
else:
    imgs_train, imgs_name_train, classes_train = read_img(train=True)
    train_data = {
        'imgs': imgs_train,
        'imgs_name': imgs_name_train,
        'classes': classes_train
    }
    torch.save(train_data, train_data_path)
    print('first generate train data and save to %s' % train_data_path)

print(len(imgs_train))


def cv_split(num_cv, imgs_train, imgs_name_train, classes_train):
    kf = KFold(n_splits=num_cv)

    def get_sublist(arr, idx):
        sub = [arr[b] for b in idx]
        return sub
    imgs_train_cv, imgs_val_cv = [], []
    imgs_name_train_cv, imgs_name_val_cv = [], []
    classes_train_cv, classes_val_cv = [], []
    for train_index, test_index in kf.split(imgs_train):
        i_train, i_val = get_sublist(imgs_train, train_index), get_sublist(imgs_train, test_index)
        imgs_train_cv.append(i_train)
        imgs_val_cv.append(i_val)

        i_n_train, i_n_val = get_sublist(imgs_name_train, train_index), get_sublist(imgs_name_train, test_index)
        imgs_name_train_cv.append(i_n_train)
        imgs_name_val_cv.append(i_n_val)

        c_train, c_val = get_sublist(classes_train, train_index), get_sublist(classes_train, test_index)
        classes_train_cv.append(c_train)
        classes_val_cv.append(c_val)
    train_data_cv = {
        'imgs': imgs_train_cv,
        'imgs_name': imgs_name_train_cv,
        'classes': classes_train_cv
    }
    val_data_cv = {
        'imgs': imgs_val_cv,
        'imgs_name': imgs_name_val_cv,
        'classes': classes_val_cv
    }
    return train_data_cv, val_data_cv

train_data_cv, val_data_cv = cv_split(NUM_CV, imgs_train, imgs_name_train, classes_train)


def get_loader_cv(train_data_cv, val_data_cv):
    imgs_train_cv, imgs_val_cv = train_data_cv['imgs'], val_data_cv['imgs']
    imgs_name_train_cv, imgs_name_val_cv = train_data_cv['imgs_name'], val_data_cv['imgs_name']
    classes_train_cv, classes_val_cv = train_data_cv['classes'], val_data_cv['classes']
    train_loader_cv, val_loader_cv = [], []
    for i in range(len(imgs_train_cv)):
        i_train, i_val = imgs_train_cv[i], imgs_val_cv[i]
        i_n_train, i_n_val = imgs_name_train_cv[i], imgs_name_val_cv[i]
        c_train, c_val = classes_train_cv[i], classes_val_cv[i]
        train_set = DogImageLoader(data_type='all', imgs=i_train, imgs_name=i_n_train,
                                   classes=c_train, transform=img_transform_299['train'])
        val_set = DogImageLoader(data_type='all', imgs=i_val, imgs_name=i_n_val,
                                 classes=c_val, transform=img_transform_299['val'])
        print(len(train_set), len(val_set))
        t_loader = data_utils.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        v_loader = data_utils.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        train_loader_cv.append(t_loader)
        val_loader_cv.append(v_loader)
        return train_loader_cv, val_loader_cv


train_loader_cv, val_loader_cv = get_loader_cv(train_data_cv, val_data_cv)

model_name_all_cv = ['densenet161_cv_%i' % c for c in range(1, NUM_CV+1)]
for i, in [0, 2, 4, 6, 8]:
    train_loader = train_loader_cv[i]
    val_loader = val_loader_cv[i]
    model_i, sgd_i = model_cv[i], sgd_cv[i]
    model_name = model_name_all_cv[i]
    for epoch in range(93):
        train_epoch_cv(epoch, model_name, model_i, train_loader, sgd_i, cross_entropy)
        test_epoch_cv(epoch, model_name, model_i, val_loader, cross_entropy)
    model_i.cpu()


