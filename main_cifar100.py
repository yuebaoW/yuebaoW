'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from torchstat import stat
from tensorboardX import SummaryWriter

writer = SummaryWriter('./log')

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=8)
#
# classes = ('plane', 'car', 'bird', 'cat', 'deer',
           # 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

List_Net = {"VGG('VGG19')": VGG('VGG19'),
            "ResNet18()": ResNet18(),
            'PreActResNet18()': PreActResNet18(),
            "GoogLeNet()": GoogLeNet(),
            "DenseNet121()": DenseNet121(),
            "ResNeXt29_2x64d()": ResNeXt29_2x64d(),
            "MobileNet()": MobileNet(),
            "MobileNetV2()": MobileNetV2(),
            "DPN92()": DPN92(),
            # "ShuffleNetG2()": ShuffleNetG2(),
            "SENet18()": SENet18(),
            "ShuffleNetV2(1)": ShuffleNetV2(1),
            "EfficientNetB0()": EfficientNetB0(),
            "RegNetX_200MF()": RegNetX_200MF(),
            "SimpleDLA()": SimpleDLA()
            }

for key in List_Net:
    net = List_Net[key]

    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()

    # print('==> Displaying model..')

    net = net.to(device)

    # b = stat(net, (3, 32, 32))
    # a = input()
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # Training

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Model: %s'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total, key))

        writer.add_scalars('Train_ACCURACY', {key + '_Train_ACC': correct * 100. / total}, epoch)

        writer.add_scalars('Train_LOSS FUNCTION', {key + '_Train_LOSS': train_loss / len(trainloader) - 1}, epoch)


    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Model: %s '
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total, key))

            writer.add_scalars('Test_ACCURACY', {key + '_Test_ACC': correct * 100. / total}, epoch)

            writer.add_scalars('Test_LOSS FUNCTION', {key + '_Test_LOSS': test_loss / len(testloader) - 1}, epoch)


    # Save checkpoint.
    # acc = 100. * correct / total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.pth')
    #     best_acc = acc

    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch)
        test(epoch)
        scheduler.step()

    print(key+"Finished")
    torch.cuda.empty_cache()
    print("Memory emptying")

