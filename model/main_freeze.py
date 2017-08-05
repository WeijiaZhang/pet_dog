from __future__ import print_function
import torch
import torchvision

import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

from train_val import *
from data_preprocess.img_process import *

train_data_path = '../processed/train_val10_299_1.t7'
state_path_152 = '../results/best_params/res152_freeze_79_ckpt.t7'

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

# model_inc_v3 = models.inception_v3(pretrained=True)
# num_ftrs = model_inc_v3.fc.in_features
# model_inc_v3.fc = nn.Linear(num_ftrs, NUM_CLASSES)
# model_inc_v3 = nn.DataParallel(model_inc_v3, device_ids=device_id_all)

model_dense161 = models.densenet161(pretrained=True)
num_ftrs = model_dense161.classifier.in_features
model_dense161.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
fc_params_dense161 = model_dense161.classifier.parameters()
model_dense161 = nn.DataParallel(model_dense161, device_ids=device_id_all)


def load_model(model, state_dict_path):
    state = torch.load(state_dict_path)
    m_name = state['model_name']
    state_dict = state['model_param']
    e = state['epoch']
    model.load_state_dict(state_dict)
    print(e, m_name)
    return model

# model_res152 = load_model(model_res152, state_path_152)

if use_cuda:
    # model_inc_v3.cuda()
    # model_res101.cuda()
    model_res152.cuda()
    model_dense161.cuda()

cross_entropy = nn.CrossEntropyLoss()
# sgd_res101 = optim.SGD(model_res101.parameters(), lr=3e-4, momentum=0.9, weight_decay=5e-3)
# sgd_inc_v3 = optim.SGD(model_inc_v3.parameters(), lr=3e-4, momentum=0.9, weight_decay=5e-3)


# sgd_res152 = optim.SGD(filter(lambda p: p.requires_grad, model_res152.parameters()),
#                        lr=1e-3, momentum=0.9, weight_decay=5e-3)

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


model_name_all = ['resnet101', 'resnet152', 'densenet161', 'inception_v3', 'vgg16', 'alexnet']
for epoch in range(100):
    model_name = model_name_all[2]
    train_epoch(epoch, model_name, model_dense161, train_loader, fc_params_dense161, cross_entropy)
    test_epoch(epoch, model_name, model_dense161, val_loader, cross_entropy)
model_dense161.cpu()
for epoch in range(100):
    model_name = model_name_all[1]
    train_epoch(epoch, model_name, model_res152, train_loader, fc_params_res152, cross_entropy)
    test_epoch(epoch, model_name, model_res152, val_loader, cross_entropy)
model_res152.cpu()


