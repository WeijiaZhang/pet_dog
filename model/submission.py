from __future__ import  print_function

import pandas as pd
import sklearn.metrics as metrics
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from data_preprocess.img_process import *

import matplotlib.pyplot as plt
import seaborn as sn


state_path_101 = '../results/best_params/res101_73_ckpt.t7'

state_path_152 = '../results/best_params/res152_91_ckpt.t7'

state_path_v3 = '../results/best_params/inception_v3_88_ckpt.t7'

device_id_all = [0, 1, 2, 3]
use_cuda = torch.cuda.is_available()

# load best resnet101
model_res101 = models.resnet101()
num_ftrs = model_res101.fc.in_features
model_res101.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model_res101 = nn.DataParallel(model_res101, device_ids=device_id_all)

state_101 = torch.load(state_path_101)
model_name_101 = state_101['model_name']
state_dict_101 = state_101['model_param']
model_res101.load_state_dict(state_dict_101)
print(model_name_101)

# load best resnet152
model_res152 = models.resnet152()
num_ftrs = model_res152.fc.in_features
model_res152.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model_res152 = nn.DataParallel(model_res152, device_ids=device_id_all)

state_152 = torch.load(state_path_152)
model_name_152 = state_152['model_name']
state_dict_152 = state_152['model_param']
model_res152.load_state_dict(state_dict_152)
print(model_name_152)

# load best inception v3
model_inc_v3 = models.inception_v3()
num_ftrs = model_inc_v3.fc.in_features
model_inc_v3.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model_inc_v3 = nn.DataParallel(model_inc_v3, device_ids=device_id_all)

state_v3 = torch.load(state_path_v3)
model_name_v3 = state_v3['model_name']
state_dict_v3 = state_v3['model_param']
model_inc_v3.load_state_dict(state_dict_v3)
print(model_name_v3)

if use_cuda:
    model_res101.cuda()
    model_res152.cuda()

img_transform = img_transform['val']
val_set = DogImageLoader(data_type='val', transform=img_transform['val'])
val_loader = data_utils.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

test_set = DogImageLoader(data_type='test', transform=img_transform['val'])

test_loader = data_utils.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=4)
print(len(val_set), len(test_set))


def validation(model, data_loader, criterion):
    model.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    total = 0
    pred_all = np.empty(0)
    true_all = np.empty(0)
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs, volatile=True), Variable(labels)
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).data[0]
        # get the index of the max log-probability
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        true_all = np.concatenate((true_all, labels.data.cpu().numpy().reshape(-1)))
        pred_all = np.concatenate((pred_all, predicted.cpu().numpy().reshape(-1)))
    test_loss /= len(data_loader)
    test_acc = 100.*correct / total
    test_err = 100. - test_acc
    true_all = np.array(true_all).reshape(-1, 1)
    pred_all = np.array(pred_all).reshape(-1, 1)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%, Error: {:.2f}%'.format(test_loss, test_acc, test_err))
    return true_all, pred_all


def predict(model, data_loader):
    model.eval()
    use_cuda = torch.cuda.is_available()
    pred_all = np.empty(0)
    for inputs, labels in data_loader:
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs, volatile=True), Variable(labels)
        outputs = model(inputs)
        # get the index of the max log-probability
        _, predicted = torch.max(outputs.data, 1)
        pred_all = np.concatenate((pred_all, predicted.cpu().numpy().reshape(-1)))
    pred_all = np.array(pred_all).reshape(-1, 1)
    return pred_all

cs = nn.CrossEntropyLoss()
for i in range(3):
    y_true, y_pred = validation(model_res152, val_loader, cs)
    # confu_mat = metrics.confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(40, 40))
    # sn.heatmap(confu_mat, annot=True)
    # plt.show()
test_pred = predict(model_res152, test_loader)


def format_pred(test_pred):
    out = pd.DataFrame(columns=['img_label', 'img_id'])
    imgs_name = test_set.get_imgs_name()
    print(test_pred.shape, len(imgs_name))
    for i, name in enumerate(imgs_name):
        pred_i = int(test_pred[i])
        data = {'img_label': pred_i, 'img_id': name}
        out = out.append(data, ignore_index=True)
    return out

output_df = format_pred(test_pred)
output_df.to_csv('../results/base_%s.txt' % model_name_152, sep='\t', header=None, index=False)







