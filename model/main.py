from __future__ import print_function
import torch
import torchvision

import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

from train_val import *
from data_preprocess.img_process import *

use_cuda = torch.cuda.is_available()
device_id = 2
device_id_part1 = [0, 1]
device_id_part2 = [2, 3]
device_id_all = [0, 1, 2, 3]

model_res101 = models.resnet101(pretrained=True)
num_ftrs = model_res101.fc.in_features
model_res101.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model_res101 = nn.DataParallel(model_res101, device_ids=device_id_all)

model_res152 = models.resnet152(pretrained=True)
num_ftrs = model_res152.fc.in_features
model_res152.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model_res152 = nn.DataParallel(model_res152, device_ids=device_id_all)

model_inc_v3 = models.inception_v3(pretrained=True)
num_ftrs = model_inc_v3.fc.in_features
model_inc_v3.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model_inc_v3 = nn.DataParallel(model_inc_v3, device_ids=device_id_all)

if use_cuda:
    model_res101.cuda()
    model_res152.cuda()
    model_inc_v3.cuda()

cross_entropy = nn.CrossEntropyLoss()
sgd_res101 = optim.SGD(model_res101.parameters(), lr=3e-4, momentum=0.9, weight_decay=5e-3)

sgd_res152 = optim.SGD(model_res152.parameters(), lr=3e-4, momentum=0.9, weight_decay=5e-3)

sgd_inc_v3 = optim.SGD(model_inc_v3.parameters(), lr=3e-4, momentum=0.9, weight_decay=5e-3)

train_set = DogImageLoader(data_type='train', transform=img_transform['train'])
val_set = DogImageLoader(data_type='val', transform=img_transform['val'])

# all_classes = all_set.get_classes()
# train_classes = train_set.get_classes()
# val_calsses = val_set.get_classes()
# print(len(train_set), len(val_set))
# print(len(np.unique(all_classes)), len(np.unique(train_classes)), len(np.unique(val_calsses)))
# print(np.unique(all_classes))

train_loader = data_utils.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = data_utils.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


model_name_all = ['res101', 'res152', 'inception_v3', 'vgg16', 'alexnet']
for epoch in range(300):
    model_name = model_name_all[1]
    train_epoch(epoch, model_name, model_res152, train_loader, sgd_res152, cross_entropy)
    test_epoch(epoch, model_name, model_res152, val_loader, cross_entropy)


