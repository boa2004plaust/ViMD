import os
import torch
import random
import numpy as np
from collections import defaultdict
from torchvision import transforms, datasets
from torchvision.transforms.functional import rotate
from torch.utils.data import Sampler, BatchSampler, RandomSampler, DataLoader
from MyImageFolder import MyDataset

aug_size = 224 // 7  # 224
org_size = 224 + aug_size
src_size = 224
dst_size = 56
transform_so_train = transforms.Compose(
    [
        transforms.RandomResizedCrop((src_size, src_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
    ])
transform_so_test = transforms.Compose(
    [
        transforms.Resize(src_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((src_size, src_size))
    ])
transform_mo = transforms.Compose(
    [
        transforms.Resize((dst_size, dst_size), interpolation=transforms.InterpolationMode.BICUBIC),
    ])
transform_eo = transforms.Compose(
    [
        transforms.ToTensor()
    ])


def data_load(batch_size, data_dir, data):
    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'val')
    if data.lower() == 'cub':
        num_label = 200
    elif data.lower() == 'car':
        num_label = 196
    elif data.lower() == 'air':
        num_label = 100
    elif data.lower() == 'pet':
        num_label = 37
    elif data.lower() == 'flowers':
        num_label = 102
    elif data.lower() == 'mit67':
        num_label = 67
    elif data.lower() == 'dog':
        num_label = 120
    else:
        raise NotImplementedError('Dataset {} is not prepared.'.format(data))
    kwargs = {'num_workers': 4, 'pin_memory': True}
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    trainset = MyDataset(train_path, transform_s=transform_so_train,
                         transform_e=transform_eo, transform_m=transform_mo)

    testset = MyDataset(test_path, transform_s=transform_so_test,
                        transform_e=transform_eo, transform_m=transform_mo)

    return trainset, testset, num_label
