"""
Initial setup based on https://github.com/kuangliu/pytorch-cifar
and https://github.com/weiaicunzai/pytorch-cifar100
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np

import os
import argparse
from tqdm import tqdm

# Training
from bias_transfer.data import *
from bias_transfer.models.resnet import *


def train(net, data_loader, device, epoch, criterion, optimizer, add_noise=False, mean=0.0, std=1.0):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    with tqdm(data_loader) as t:
        for batch_idx, (inputs, targets) in enumerate(t):
            inputs, targets = inputs.to(device), targets.to(device)
            if add_noise:
                with torch.no_grad():
                    inputs += torch.normal(mean=mean, std=std, size=inputs.size()).cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            t.set_postfix(acc=100. * correct / total, loss=loss.item())


def test(net, data_loader, device, epoch, criterion, best_acc, add_noise=False, mean=0.0, std=1.0):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(data_loader) as t:
            if add_noise:
                torch.manual_seed(42)  # so that we always have the same noise for evaluation!
            for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader)):
                inputs, targets = inputs.to(device), targets.to(device)
                if add_noise:
                    inputs += torch.normal(mean=mean, std=std, size=inputs.size()).cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                t.set_postfix(acc=100. * correct / total, loss=loss.item())

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    return best_acc


def main(force_cpu: bool = False, lr: float = 0.1, resume: bool = False, cifar100: bool = True, batch_size: int = 128,
         add_noise: bool = False, mean: float = 0.0, std: float = 1.0):
    device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    print('==> Load data..')
    if cifar100:
        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        train_loader = get_training_data_loader(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD, batch_size=batch_size)
        test_loader = get_test_data_loader(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD, batch_size=batch_size)
    else:  # CIFAR10
        CIFAR10_TRAIN_MEAN = (0.49139968, 0.48215841, 0.44653091)
        CIFAR10_TRAIN_STD = (0.24703223, 0.24348513, 0.26158784)
        train_loader = get_training_data_loader(mean=CIFAR10_TRAIN_MEAN, std=CIFAR10_TRAIN_STD, cifar100=False,
                                                batch_size=batch_size)
        test_loader = get_test_data_loader(mean=CIFAR10_TRAIN_MEAN, std=CIFAR10_TRAIN_STD, cifar100=False,
                                           batch_size=batch_size)

    # Model
    print('==> Building model..')
    net = resnet50(num_classes=100 if cifar100 else 10)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        # if add_noise:
        #     print("moving to gpu")
        #     mean = torch.scalar_tensor(mean).cuda()
        #     std = torch.scalar_tensor(std).cuda()

    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160],
                                                     gamma=0.2)  # learning rate decay

    for epoch in range(start_epoch, start_epoch + 200):
        train(net=net, data_loader=train_loader, device=device, epoch=epoch, criterion=criterion, optimizer=optimizer,
              add_noise=add_noise, mean=mean, std=std)
        test(net=net, data_loader=test_loader, device=device, epoch=epoch, criterion=criterion, best_acc=best_acc,
             add_noise=add_noise, mean=mean, std=std)
        train_scheduler.step(epoch=epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    main(force_cpu=False, lr=args.lr, resume=args.resume, cifar100=False)
