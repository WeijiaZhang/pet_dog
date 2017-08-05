from __future__ import  print_function

import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from data_preprocess.img_process import *

import matplotlib.pyplot as plt
import seaborn as sn


# state_path_101 = '../results/best_params/res101_73_ckpt.t7'
state_path_v3 = '../results/best_params/inception_v3_80_ckpt.t7'

state_path_161 = '../results/best_params/densenet161_80_ckpt.t7'
state_path_152 = '../results/best_params/resnet152_80_ckpt.t7'

train_data_path = '../processed/train_val10_299_1.t7'

device_id_all = [0, 1, 2, 3]
use_cuda = torch.cuda.is_available()

# load best resnet101
# model_res101 = models.resnet101()
# num_ftrs = model_res101.fc.in_features
# model_res101.fc = nn.Linear(num_ftrs, NUM_CLASSES)
# model_res101 = nn.DataParallel(model_res101, device_ids=device_id_all)

# load best resnet152
model_res152 = models.resnet152()
num_ftrs = model_res152.fc.in_features
model_res152.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model_res152 = nn.DataParallel(model_res152, device_ids=device_id_all)


# load best inception v3
model_inc_v3 = models.inception_v3()
num_ftrs = model_inc_v3.fc.in_features
model_inc_v3.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model_inc_v3 = nn.DataParallel(model_inc_v3, device_ids=device_id_all)

# load best resnet161
model_dense161 = models.densenet161()
num_ftrs = model_dense161.classifier.in_features
model_dense161.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
model_dense161 = nn.DataParallel(model_dense161, device_ids=device_id_all)


def load_model(model, state_path):
    state = torch.load(state_path)
    model_name = state['model_name']
    state_dict = state['model_param']
    best_acc = state['test_best_acc']
    epoch = state['epoch']
    model.load_state_dict(state_dict)
    print(epoch, model_name)
    print('best accuracy: %.2f' % best_acc)
    return model_name, model

model_name_v3, model_inc_v3 = load_model(model_inc_v3, state_path_v3)

if use_cuda:
    # model_res101.cuda()
    model_inc_v3.cuda()
    # model_res152.cuda()
    # model_dense161.cuda()

imgs_train, imgs_name_train, classes_train = [], [], []
# read train dataset
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
imgs_test, imgs_name_test, classes_test = read_img(train=False)


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


def get_val_loader(cv_idx, val_data_cv):
    i = cv_idx
    imgs_train_cv, imgs_val_cv = train_data_cv['imgs'], val_data_cv['imgs']
    imgs_name_train_cv, imgs_name_val_cv = train_data_cv['imgs_name'], val_data_cv['imgs_name']
    classes_train_cv, classes_val_cv = train_data_cv['classes'], val_data_cv['classes']
    i_train, i_val = imgs_train_cv[i], imgs_val_cv[i]
    i_n_train, i_n_val = imgs_name_train_cv[i], imgs_name_val_cv[i]
    c_train, c_val = classes_train_cv[i], classes_val_cv[i]
    val_set = DogImageLoader(data_type='all', imgs=i_val, imgs_name=i_n_val,
                             classes=c_val, transform=img_transform_256['val'])
    val_loader = data_utils.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print('val_set: %i' % len(val_set))
    return val_loader


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
train_data_cv, val_data_cv = cv_split(NUM_CV, imgs_train, imgs_name_train, classes_train)
test_set = DogImageLoader(data_type='test', imgs=imgs_test, imgs_name=imgs_name_test,
                          classes=classes_test, transform=img_transform_256['val'])
print('test_set: %i' % len(test_set))
test_loader = data_utils.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=4)
for i in range(3):
    val_loader = get_val_loader(i, val_data_cv)
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
output_df.to_csv('../results/base_%s.txt' % model_name_v3, sep='\t', header=None, index=False)







