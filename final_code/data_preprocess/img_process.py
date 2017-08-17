import os
import pandas as pd
import numpy as np
import random
import torch
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms

from config import *
from PIL import Image
import matplotlib.pyplot as plt


def default_loader(path):
    return Image.open(path).convert('RGB')


def read_img(train=True):
    imgs = []
    imgs_name = []
    classes = []
    if train:
        train_img_label = open(TRAIN_LABEL_PATH)
        for line in train_img_label.readlines():
            la = line.split()
            img_name = la[0] + '.jpg'
            img_class_idx = int(la[1])
            if os.path.isfile(os.path.join(TRAIN_IMG_PATH, img_name)):
                imgs_name.append(la[0])
                classes.append(img_class_idx)
                img_name = TRAIN_IMG_PATH + img_name
                imgs.append((img_name, img_class_idx))
        val_img_label = open(VAL_LABEL_PATH)
        for line in val_img_label.readlines():
            la = line.split()
            img_name = la[0] + '.jpg'
            img_class_idx = int(la[1])
            if os.path.isfile(os.path.join(VAL_IMG_PATH, img_name)):
                imgs_name.append(la[0])
                classes.append(img_class_idx)
                img_name = VAL_IMG_PATH + img_name
                imgs.append((img_name, img_class_idx))
    else:
        # that is train = False
        test_img_label = open(TEST_LABEL_PATH)
        for line in test_img_label.readlines():
            la = line.split()
            img_name = la[0] + '.jpg'
            img_class_idx = int(la[1])
            if os.path.isfile(os.path.join(TEST_IMG_PATH, img_name)):
                imgs_name.append(la[0])
                classes.append(img_class_idx)
                img_name = TEST_IMG_PATH + img_name
                imgs.append((img_name, img_class_idx))
    return imgs, imgs_name, classes


def read_augment_img(data_path, train=True):
    data_path = '../' + data_path
    print(data_path)
    data_df = pd.read_csv(data_path, header=None, sep='\t', names=['img', 'label'])
    if train:
        root = '../' + TRAIN_IMG_AUG_PATH
    else:
        root = '../' + TEST_IMG_AUG_PATH
    for i, data_path in enumerate(data_df['img'].values):
        if i > 2:
            break
        data_path = '../' + data_path
        img_path, img_name = os.path.split(data_path)
        img = default_loader(data_path)
        img_flip_lr = img.transpose(Image.FLIP_LEFT_RIGHT)
        # img_flip_tb = img.transpose(Image.FLIP_TOP_BOTTOM)

        img_aug_all = []
        for degree in [0]:
            img_aug_all.append(img.rotate(degree))
            img_aug_all.append(img_flip_lr.rotate(degree))

        for i, img_aug in enumerate(img_aug_all):
            save_path = root + img_name[:-4] + '_%i' % i + img_name[-4:]
            img_aug.save(save_path)


def make_dataset(data_path=None, train=True):
    if train:
        imgs_train, imgs_val = [], []
        train_val_file = np.load(data_path)
        imgs_name_train = train_val_file['imgs_train']
        labels_train = train_val_file['labels_train']
        imgs_name_val = train_val_file['imgs_val']
        labels_val = train_val_file['labels_val']
        for i, i_train in enumerate(imgs_name_train):
            la_train = labels_train[i]
            imgs_train.append((i_train, la_train))
        for i, i_val in enumerate(imgs_name_val):
            la_val = labels_val[i]
            imgs_val.append((i_val, la_val))
        return imgs_train, imgs_val
    else:
        imgs = []
        test_img_label = open(data_path)
        for line in test_img_label.readlines():
            la = line.split()
            img_class_idx = int(la[0])
            img_name = la[1] + '.jpg'
            if os.path.isfile(os.path.join(TEST_IMG_PATH, img_name)):
                img_name = TEST_IMG_PATH + img_name
                imgs.append((img_name, img_class_idx))
        return imgs


class DogImageLoader(data_utils.Dataset):
    def __init__(self, root=None, imgs=None, transform=None,
                 target_transform=None, loader=default_loader):
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        img_name, target = self.imgs[index]
        if self.root is None:
            img_path = img_name
        else:
            img_path = self.root+img_name
        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


def my_imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        print(repr(title).decode('unicode-escape'))
        plt.title(repr(title).decode('unicode-escape'))
    plt.show()

normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def tr_crop_flip(img_size):
    img_transform_train = transforms.Compose(
        [transforms.RandomSizedCrop(img_size),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         normalize]
    )
    return img_transform_train


def tr_scale_crop(img_size):
    img_transform_val = transforms.Compose(
        [transforms.Scale(img_size),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         normalize])
    return img_transform_val


def tr_scale(img_size):
    img_transform_val = transforms.Compose(
        [transforms.Scale(img_size),
         transforms.ToTensor(),
         normalize])
    return img_transform_val

# classes_name = train_set.get_name()
# inputs, targets = next(iter(train_loder))
# out = torchvision.utils.make_grid(inputs, nrow=5)
# my_imshow(out, title=[classes_name[int(x)] for x in targets])


if __name__ == '__main__':
    read_augment_img(TRAIN_VAL_PATH)
