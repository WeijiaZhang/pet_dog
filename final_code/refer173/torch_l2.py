# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import os, sys, h5py, gc, argparse, codecs, shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from torch_model import *


train_val_path = '../../dataset/train/train_val_data.txt'

device_id = 2
num_classes = 134
model_name_all = ['ensemble_1789', 'ensemble_load_cv_3', 'ensemble_load_cv_5', 'ensemble_load_cv_7', 'vgg_inception_all', 'resnet_all', 'densenet_all']

idx_cv = 3
model_name = 'ensemble_load_cv_%i' % idx_cv

#train = pd.read_csv('../../dataset/train/data_train_image.txt', header=None, sep=' ', names=['img', 'label', 'url'])
#val = pd.read_csv('../../dataset/train/val.txt', header=None, sep=' ', names=['img', 'label', 'url'])

# train['img'] = '../input/train/' + train['img'] + '.jpg'
# val['img'] = '../input/test1/' + val['img'] + '.jpg'

train_val = pd.read_csv(train_val_path, header=None, sep='\t', names=['img', 'label'])
train_val['img'] = '../' +  train_val['img']
train_img_name = []
for i_name in train_val['img'].values:
    name = i_name.split('/')[-1]
    name = name.split('.')[0]
    train_img_name.append(name)

# 删除标签不一致的情况
#train = train[~train['img'].duplicated(keep=False)]
#train_val = pd.concat([train, val], axis=0, ignore_index=True)

# lbl = LabelEncoder()
# train_val['label'] = lbl.fit_transform(train_val['label'].values)

# train_val = train_val.iloc[: 1000]
#test = os.listdir('../../dataset/test/test_image/')
#test = [x[: -4] for x in test]

test_path = '../../dataset/test/test_info.txt'
test_df = pd.read_csv(test_path, header=None, sep='\t', names=['img', 'label'])
test = test_df['img'].values

train_feat, test_feat = [], []
feature_file = [
    # './feature/googlenet_pet_breed.h5',

    # './feature_yolo/resnet18.h5',
    # './feature_yolo/densenet161.h5',
    # './feature_yolo/densenet169.h5',
    # './feature_yolo/densenet201.h5',
    # './feature_yolo/densenet121.h5',
    # './feature_yolo/densenet201.h5',

    # 'dpn92.h5'

    # './feature/vgg11_bn.h5',
    # './feature/vgg13_bn.h5',
    # './feature/vgg16_bn.h5',
    # './feature/vgg19_bn.h5',

    # './feature/resnet18.h5',
    # './feature/resnet34.h5',
     './feature/load_cv_%i/resnet50_load.h5'% idx_cv,
     './feature/load_cv_%i/resnet101_load.h5'% idx_cv,
     './feature/load_cv_%i/resnet152_load.h5'% idx_cv,

    # './feature/densenet121.h5',
     './feature/load_cv_%i/densenet161_load.h5'% idx_cv,
     './feature/load_cv_%i/densenet169_load.h5'% idx_cv,
    # './feature/densenet201.h5',

     './feature/load_cv_%i/inception_v3_load.h5'% idx_cv,
]
for ffile in feature_file:
    with h5py.File(ffile, "r") as f:
        train_feat.append(f['train_feature'][:])
        test_feat.append(f['test_feature'][:])

train_feat = np.concatenate(train_feat, 1)
test_feat = np.concatenate(test_feat, 1)
print(feature_file)
print('Train Feature: ', train_feat.shape)
print('Test Feature: ', test_feat.shape)


class modelnn(nn.Module):
    def __init__(self):
        super(modelnn, self).__init__()
        self.model = nn.Sequential(
            #nn.Dropout(0.05),
            nn.Linear(train_feat.shape[1], 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), train_feat.shape[1])
        out = self.model(x)
        return out


def adjust_learning_rate(optimizer, epoch, decay_epoch=40,
			 init_lr=1e-4, init_wd=5e-4):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.6 ** (epoch // decay_epoch))
    wd = init_wd * (0.6 ** (epoch // decay_epoch))
    
    if epoch % decay_epoch == 0:
	print('lr is set to {}'.format(lr))
	print('weight decay is set to {}'.format(wd))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
	param_group['weight_decay'] = wd

epoch_size = 100
batch_size = 128
skf = KFold(n_splits=10)
train_preds, test_preds = np.zeros(train_feat.shape[0]), []
test_prob_all = []
train_logs = [[], [], [], []]
for i, (train_index, test_index) in enumerate(skf.split(train_feat)):
    if i != (idx_cv-1):
	continue
    # that i is idx_cv - 1
    print('%i: min test index: %i, max test index: %i'% (i, min(test_index), max(test_index)))
    X_train, X_test = train_feat[train_index, :], train_feat[test_index, :]
    y_train, y_test = train_val['label'].values[train_index], train_val['label'].values[test_index]

    # from imblearn.over_sampling import SMOTE
    # sm = SMOTE()
    # X_train, y_train = sm.fit_sample(X_train, y_train)

    train_set = Arrayloader(X_train, y_train)
    train_loader = data_utils.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    val_set = Arrayloader(X_test, y_test)
    val_loader = data_utils.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)

    model = modelnn()
    model = model.cuda(device_id)

    class_weight = np.array([1.65876777, 1.4, 1.45228216, 1.47058824, 1.4,
                             1.89189189, 1.60550459, 4.72972973, 1.75879397, 4.86111111,
                             4.16666667, 7., 1.40562249, 6.48148148, 3.5,
                             6.03448276, 1.75, 1.75, 1.40562249, 4.48717949,
                             1.41129032, 1.96629213, 1.77664975, 1.4, 1.69082126,
                             1.4, 5.2238806, 1.4, 1.41129032, 1.4,
                             2.09580838, 3.27102804, 1.42276423, 1.4, 3.72340426,
                             1.40562249, 1.4, 1.40562249, 1.4, 3.39805825,
                             5.14705882, 1.44032922, 3.36538462, 5.73770492, 2.77777778,
                             2.09580838, 1.40562249, 1.41129032, 1.4, 1.40562249,
                             1.40562249, 2.51798561, 1.4, 1.97740113, 1.41700405,
                             1.54867257, 1.79487179, 1.66666667, 2.71317829, 1.89189189,
                             1.40562249, 1.41129032, 1.4, 1.41129032, 1.40562249,
                             3.24074074, 1.60550459, 2.13414634, 2.65151515, 3.80434783,
                             2.1875, 2.04678363, 1.40562249, 2.86885246, 3.72340426,
                             1.66666667, 1.40562249, 2.71317829, 4.32098765, 3.64583333,
                             3.18181818, 5.73770492, 1.63551402, 1.2195122, 6.60377358,
                             1.52173913, 5.07246377, 1.40562249, 1.88172043, 1.41700405,
                             1.40562249, 1.40562249, 1.4, 1.4, 1.54867257,
                             1., 2.69230769, 1.2962963, 1.4, 6.25], dtype=np.float32)
    class_weight = torch.from_numpy(class_weight)
    # criterion = nn.CrossEntropyLoss(weight = class_weight.float()).cuda()
    criterion = nn.CrossEntropyLoss().cuda(device_id)
    optimizer_ft = torch.optim.SGD(model.parameters(), lr=2e-4, momentum=0.9, weight_decay=5e-3)

    for epoch in range(epoch_size):
	model.train()
        adjust_learning_rate(optimizer_ft, epoch)
        # Traing batch
        running_corrects = 0.0
        running_loss = 0.0
        for data in train_loader:
            dta_x, dta_y = data
            dta_x, dta_y = Variable(dta_x.cuda(device_id)), Variable(dta_y.cuda(device_id).view(dta_y.size(0)))
            optimizer_ft.zero_grad()
            outputs = model(dta_x)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, dta_y)
            loss.backward()
            optimizer_ft.step()

            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == dta_y.data)

        train_loss = running_loss / len(train_set)
        train_acc = running_corrects / len(train_set) * 100.0

        # Val batch
	model.eval()
        running_corrects = 0.0
        running_loss = 0.0
	best_acc_path = '../../checkpoint/best_acc_cv/best_acc_%s_cv_%i.t7'% (model_name, i)
        for data in val_loader:
            dta_x, dta_y = data
            dta_x, dta_y = Variable(dta_x.cuda(device_id)), Variable(dta_y.cuda(device_id).view(dta_y.size(0)))
            outputs = model(dta_x)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, dta_y)

            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == dta_y.data)

        val_loss = running_loss / len(val_set)
        val_acc = running_corrects / len(val_set) * 100.0
	if os.path.isfile(best_acc_path):
	    temp = torch.load(best_acc_path)
	    val_best_acc = temp['test_best_acc']
	else:
	    val_best_acc = 0.0
	if val_acc > val_best_acc:
	    val_best_acc = val_acc
	    temp = {'test_best_acc': val_best_acc}
	    torch.save(temp, best_acc_path)
	    if val_best_acc > 80:
	        print('Saving %s of epoch %i...' % (model_name, epoch+1))
		state = {
		    'model_name': model_name + '_' + str(val_best_acc),
		    'model_param': model.state_dict(),
		    'test_best_acc': val_best_acc,
		    'epoch': epoch
		}
		torch.save(state, '../../checkpoint/%s/%s_cv_%i_%i_ckpt.t7' % (model_name, model_name, i, val_best_acc))
	    print('{}: best accuracy {:.2f}%'.format(model_name, val_best_acc))
	    
        epoch_log = '[{}/{}] | (Train/Val) Loss: {:.4f} / {:.4f} | (Train/Val) Acc {:.2f}% / {:.2f}%'.format(
	epoch+1, epoch_size, train_loss, val_loss, train_acc, val_acc)
        print(epoch_log)
        # loging(epoch_log, './log/resnet50.log', epoch % 5 == 0)

    # load best model
    model.eval()
    if os.path.isfile(best_acc_path):
        temp = torch.load(best_acc_path)
        test_best_acc = temp['test_best_acc']
	state_path = '../../checkpoint/%s/%s_cv_%i_%i_ckpt.t7' % (model_name, model_name, i, test_best_acc)
	if os.path.isfile(state_path):
	    print('\nLoad best {} with val acc {:.2f}%'.format(model_name, test_best_acc))
   	    state = torch.load(state_path)
    	    model.load_state_dict(state['model_param'])
    val_set = Arrayloader(X_test, y_test)
    val_loader = data_utils.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)
    val_pred = []
    for data in val_loader:
        dta_x, _ = data
        dta_x = Variable(dta_x.cuda(device_id))

        outputs = model(dta_x)
        _, preds = torch.max(outputs.data, 1)
        val_pred.append(preds.cpu().numpy()[0][0])
    train_preds[test_index] = val_pred

    train_logs[0].append(train_loss)
    train_logs[1].append(val_loss)
    train_logs[2].append(train_acc)
    train_logs[3].append(val_best_acc)
    print('Best {}---Val Acc: {:.2f}%\n'.format(model_name, sum(train_preds[test_index] == y_test) * 100.0 / len(y_test)))

    test_set = Arrayloader(test_feat, np.zeros_like(test_feat))
    test_loader = data_utils.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)
    test_pred = []
    test_prob_arr = []
    for data in test_loader:
        dta_x, _ = data
        dta_x = Variable(dta_x.cuda(device_id))
        outputs = model(dta_x)
	outputs = F.softmax(outputs)
	test_prob_arr.append(outputs.data.cpu())
        _, preds = torch.max(outputs.data, 1)
        test_pred.append(preds.cpu().numpy()[0][0])
    test_prob = torch.cat(test_prob_arr, 0)
    test_preds.append(test_pred)
    test_prob_all.append(test_prob)
    # torch.save(test_prob_all, '../../results/best_prob/test_prob_all.t7')

#    with codecs.open('../../results/test_ensemble_%.2f.txt'% test_best_acc, 'w') as f:
#        for i in range(len(test)):
#            f.write(str(test_pred[i]) + '\t' + test[i] + '\n')

# print(feature_file)
print('\nAverage Loss {:.4f} / {:.4f} | Average Acc {:.2f}% / {:.2f}%'.format(np.mean(train_logs[0]), np.mean(train_logs[1]),
                                            np.mean(train_logs[2]), np.mean(train_logs[3])))

#train_val = train_val.drop('url', axis=1)
#train_val['label'] = train_preds
with codecs.open('../../results/stacking/%s_input.txt' %model_name, 'w') as f:
    for i in range(train_preds.shape[0]):
        f.write(str(int(train_preds[i])) + '\t' + train_img_name[i] + '\n')

# saving prob averaging ensemble model
print('Saving bagging ensembling prob and pred...\n')
for i, test_prob_i in enumerate(test_prob_all):
    if i == 0:
	test_prob_mean = test_prob_i
    else:
	test_prob_mean += test_prob_i
test_prob_mean /= len(test_prob_all)
torch.save(test_prob_mean, '../../results/best_prob/%s_prob_mean.t7' %model_name)
_, pred_max = torch.max(test_prob_mean, 1)
pred_max = pred_max.numpy()
with codecs.open('../../results/stacking/%s_output.txt' %model_name, 'w') as f:
    for i in range(len(test)):
	f.write(str(pred_max[i][0]) + '\t' + test[i] + '\n')

# saving voting ensemble model
# from scipy.stats import mode

# test_preds = np.array(test_preds)
# with codecs.open('../../results/%s_vote_2.txt' %model_name, 'w') as f:
#    for i in range(test_preds.shape[1]):
#        f.write(str(mode(test_preds[:, i])[0][0]) + '\t' + test[i] + '\n')
