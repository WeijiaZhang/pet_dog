from __future__ import print_function
import torch
import torchvision

import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import dpn
import inception_v4

from train_val import *
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

model_res152 = models.resnet152(pretrained=True)
num_ftrs = model_res152.fc.in_features
model_res152.fc = nn.Linear(num_ftrs, NUM_CLASSES)
fc_params_res152 = model_res152.fc.parameters()
model_res152 = nn.DataParallel(model_res152, device_ids=device_id_all)

# model_inc_v3 = models.inception_v3(pretrained=True, transform_input=True)
# model_inc_v3.aux_logits = False
# num_ftrs = model_inc_v3.fc.in_features
# model_inc_v3.fc = nn.Linear(num_ftrs, NUM_CLASSES)
# model_inc_v3 = nn.DataParallel(model_inc_v3, device_ids=device_id_all)

# model_inc_v4 = inception_v4.inceptionv4(pretrained=True, transform_input=True)
# num_ftrs = model_inc_v4.classif.in_features
# model_inc_v4.classif = nn.Linear(num_ftrs, NUM_CLASSES)
# model_inc_v4 = nn.DataParallel(model_inc_v4, device_ids=device_id_all)

model_dense161 = models.densenet161(pretrained=True)
num_ftrs = model_dense161.classifier.in_features
model_dense161.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
fc_params_dense161 = model_dense161.classifier.parameters()
model_dense161 = nn.DataParallel(model_dense161, device_ids=device_id_all)

# define dual path networks
model_dpn107 = dpn.dpn107()
num_ftrs = model_dpn107.classifier.in_features
model_dpn107.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
model_dpn107 = nn.DataParallel(model_dpn107, device_ids=device_id_all)

# define densenet201
model_dense201 = models.densenet201(pretrained=True)
num_ftrs = model_dense201.classifier.in_features
model_dense201.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
fc_params_dense201 = model_dense201.classifier.parameters()
model_dense201 = nn.DataParallel(model_dense201, device_ids=device_id_all)

if use_cuda:
    # model_inc_v3.cuda()
    # model_res101.cuda()
    # model_res152.cuda()
    # model_dense161.cuda()
    # model_dpn107.cuda()
    model_dense201.cuda()

cross_entropy = nn.CrossEntropyLoss()
# sgd_res101 = optim.SGD(model_res101.parameters(), lr=3e-4, momentum=0.9, weight_decay=5e-3)

sgd_res152 = optim.SGD(model_res152.parameters(), lr=3e-4, momentum=0.9, weight_decay=5e-3)

# sgd_inc_v3 = optim.SGD(model_inc_v3.parameters(), lr=3e-4, momentum=0.9, weight_decay=5e-3)

# sgd_inc_v4 = optim.SGD(model_inc_v4.parameters(), lr=3e-4, momentum=0.9, weight_decay=5e-3)

sgd_dense161 = optim.SGD(model_dense161.parameters(), lr=3e-4, momentum=0.9, weight_decay=5e-3)

# without pretrained
sgd_dpn107 = optim.SGD(model_dpn107.parameters(), lr=3e-3, momentum=0.9, weight_decay=5e-3)

sgd_dense201 = optim.SGD(model_dense201.parameters(), lr=3e-4, momentum=0.9, weight_decay=5e-3)

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
train_set = DogImageLoader(data_type='train', imgs=imgs_train, imgs_name=imgs_name_train,
                           classes=classes_train, transform=img_transform_299['train'])
val_set = DogImageLoader(data_type='val', imgs=imgs_train, imgs_name=imgs_name_train,
                         classes=classes_train, transform=img_transform_299['val'])

print(len(train_set), len(val_set))
# all_classes = all_set.get_classes()
# train_classes = train_set.get_classes()
# val_calsses = val_set.get_classes()
# print(len(train_set), len(val_set))
# print(len(np.unique(all_classes)), len(np.unique(train_classes)), len(np.unique(val_calsses)))
# print(np.unique(all_classes))

train_loader = data_utils.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = data_utils.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


model_name_all = ['resnet101', 'resnet152', 'inception_v3', 'inception_v4', 'densenet121', 'densenet161', 'dpn107']
# for epoch in range(100):
#     model_name = model_name_all[1]
#     train_epoch(epoch, model_name, model_res152, train_loader, sgd_res152, cross_entropy)
#     test_epoch(epoch, model_name, model_res152, val_loader, cross_entropy)
# model_res152.cpu()
for epoch in range(100):
    model_name = model_name_all[3]
    train_epoch(epoch, model_name, model_dense201, train_loader, model_dense201, cross_entropy)
    test_epoch(epoch, model_name, model_dense201, val_loader, cross_entropy)
model_dense201.cpu()


