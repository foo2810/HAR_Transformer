import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from utils.train import training
from utils.losses import ComplementEntroyLoss
from pamap2.utils import PAMAP2
from models.transformer import Transformer
from models.resnet import resnet34
from models.vgg import vgg11

def get_dataset(train_persons, test_persons, fast_channel=True):
    ret = pamap2.framing(frame_size, train_persons, activities, attributes, positions, axes)
    x_train, y_train, sub_labels, cid2act, pid2name = ret
    if fast_channel:
        x_train = np.transpose(x_train, [0, 2, 1])
    print('Train: ', cid2act)
    flg = False
    for lid in range(len(activities)):
        if lid not in y_train:
            flg = True 
            print(' >>> [Warning] Subjects(label id {}) not found in train dataset'.format(lid))
    if flg:
        raise RuntimeError('Subject are not enough.')

    ret = pamap2.framing(frame_size, test_persons, activities, attributes, positions, axes)
    x_test, y_test, sub_labels, cid2act, pid2name = ret
    if fast_channel:
        x_test= np.transpose(x_test, [0, 2, 1])
    print('Test: ', cid2act)
    flg = False
    for lid in range(len(activities)):
        if lid not in y_train:
            flg = True 
            print(' >>> [Warning] Subjects(label id {}) not found in train dataset'.format(lid))
    if flg:
        raise RuntimeError('Subject are not enough.')

    train_ds = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long))
    test_ds = torch.utils.data.TensorDataset(torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader= torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print('in_shape: {}'.format(x_test.shape[1:]))

    return train_loader, test_loader

frame_size = 256
activities = [1, 2, 3, 4, 5]
attributes = ['acc1']
positions = ['hand', 'chest', 'ankle']
axes = ['x', 'y', 'z']
all_persons = np.array([
    'subject101', 'subject102', 'subject103',
    'subject104', 'subject105', 'subject106',
    'subject107', 'subject108', #'subject109',
])
d_model = len(positions) * len(axes)
n_classes = len(activities)

n_epochs = 300
batch_size = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

# Datasets
pamap2 = PAMAP2('D:/datasets/PAMAP2/PAMAP2_Dataset/Protocol/', cache_dir='data_cache/org/')

model_list = {
    'transformer': Transformer,
    'resnet34': resnet34,
    'vgg11': vgg11,
}

param_list = {
    'transformer': {'lr': 1e-2, 'scheduler': None},
    'resnet34': {'lr': 1e-5, 'scheduler': None},
    'vgg11': {'lr': 1e-5, 'scheduler': None}
}

for test_person in all_persons:
    train_persons = all_persons[all_persons != test_person]
    test_persons = np.array([test_person])

    print('='*100)
    print('Train persons: {}'.format(train_persons))
    print('Test persons: {}'.format(test_persons))

    for model_name in model_list:
        print('[{}]'.format(model_name))
        if 'transformer' in model_name:
            model = model_list[model_name](d_model=d_model, max_seq_len=frame_size, output_dim=n_classes, n_blocks=6, n_heads=3).to(device)
        else:
            model = model_list[model_name](in_channels=d_model, num_classes=n_classes).to(device)
        param = param_list[model_name]

        if 'transformer' in model_name: flg = False 
        else: flg = True 
        train_loader, test_loader = get_dataset(train_persons, test_persons, fast_channel=flg)

        # Training
        # Adam と schedulerの相性が悪いなぜ？
    
        # Loss
        criterion = nn.CrossEntropyLoss(reduction='mean')
        # criterion = ComplementEntroyLoss()

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'])
        # optimizer = torch.optim.SGD(model.parameters(), lr=param['lr'], momentum=0.9)
        # optimizer = torch.optim.Adagrad(model.parameters(), lr=param['lr'])
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=param['lr'])

        # Optimizer scheduling
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8, 16], gamma=0.1, last_epoch=-1)
        scheduler = None

        hist = training(model, train_loader, test_loader, n_epochs, criterion, optimizer, scheduler, device=device)

        pd.DataFrame(hist).to_csv('history_{}_test-{}.csv'.format(model_name, test_person))



