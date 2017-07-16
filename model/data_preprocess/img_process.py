import os
import numpy as np
import random
import torch
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms

from data_preprocess.config import *
from PIL import Image
import matplotlib.pyplot as plt


def default_loader(path):
    return Image.open(path).convert('RGB')


class DogImageLoader(data_utils.Dataset):
    def __init__(self, data_type='train', transform=None, target_transform=None, loader=default_loader):
        imgs = []
        class_name = []
        classes = []
        imgs_name = []
        file_label = open(CLASSES_NAME)
        for line in file_label.readlines():
            la = line.split()
            name = la[0].split('---')[-1].split('|')[-1]
            name = unicode(name, 'utf-8')
            class_name.append(name)
        class_to_idx = {class_name[i]: i for i in range(len(class_name))}
        if data_type != 'test':
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
            # first shuffle
            for i in range(5):
                random.shuffle(imgs)

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
            # second shuffle
            for i in range(5):
                random.shuffle(imgs)

            # !!!keypoints: shuffle imgs(because label list is sorted)
            for i in range(5):
                random.shuffle(imgs)
            len_train = int(len(imgs) * 0.8)
            if data_type == 'train':
                self.imgs = imgs[:len_train]
                self.imgs_name = imgs_name[:len_train]
            # last 30% for validation
            if data_type == 'val':
                self.imgs = imgs[len_train:]
                self.imgs_name = imgs_name[len_train:]
        else:
            # that is data_type = 'test'
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
            self.imgs = imgs
            self.imgs_name = imgs_name
        self.class_name = class_name
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        img_name, target = self.imgs[index]
        img = self.loader(img_name)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_class_name(self):
        return self.class_name

    def get_classes(self):
        return self.classes

    def get_imgs_name(self):
        return self.imgs_name


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

img_transform_train_256 = transforms.Compose(
    [transforms.RandomSizedCrop(256),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
)

img_transform_train_299 = transforms.Compose(
    [transforms.RandomSizedCrop(299),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
)

img_transform_val_256 = transforms.Compose(
    [transforms.Scale(278),
     transforms.CenterCrop(256),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
)

img_transform_val_299 = transforms.Compose(
    [transforms.Scale(320),
     transforms.CenterCrop(299),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
)

img_transform = {
    'train': img_transform_train_299,
    'val': img_transform_val_299
}


# classes_name = train_set.get_name()
# inputs, targets = next(iter(train_loder))
# out = torchvision.utils.make_grid(inputs, nrow=5)
# my_imshow(out, title=[classes_name[int(x)] for x in targets])
