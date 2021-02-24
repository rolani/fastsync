"""
Parameter Server
================

The parameter server is a framework for distributed machine learning training.

In the parameter server framework, a centralized server (or group of server
nodes) maintains global shared parameters of a machine-learning model
(e.g., a neural network) while the data and computation of calculating
updates (i.e., gradient descent updates) are distributed over worker nodes.

"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision
import torch.optim as optim
from filelock import FileLock
import numpy as np
import time
import sys
import ray


def get_data_loader_cinic_net():
    # Data loading code
    root_dir = "/home/ubuntu/synced/aws/CINIC"

    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')

    cinic_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(train_dir,
                                         transform=transforms.Compose([transforms.ToTensor(),
                                                                       transforms.Normalize(mean=cinic_mean,
                                                                                            std=cinic_std)])),
        batch_size=16, shuffle=True)

    cinic_test = torch.utils.data.DataLoader(
        datasets.ImageFolder(val_dir, transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize(mean=cinic_mean,
                                                                              std=cinic_std)])),
        batch_size=16,
        shuffle=True)

    return cinic_train, cinic_test

def get_data_loader():
    """Safely downloads data. Returns training/validation set dataloader."""

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

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    with FileLock(os.path.expanduser("~/data.lock")):
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                "~/data",
                train=True,
                download=True,
                transform=transform_train),
            batch_size=16,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10("~/data", train=False, transform=transform_test),
            batch_size=16,
            shuffle=True)
    return train_loader, test_loader


def evaluate(model, test_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # This is only set to finish evaluation faster.
            print("Batch size is {}".format((batch_idx * len(data))))
            if batch_idx * len(data) > 128:
                break
            outputs = model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, target)

            test_loss += loss.item()
            _, pred = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
    test_loss /= total
    accuracy = 100. * correct / total

    return test_loss, accuracy



#######################################################################
# Setup: Defining the Neural Network
# ----------------------------------
#
# We define a small neural network to use in training. We provide
# some helper functions for obtaining data, including getter/setter
# methods for gradients and weights.



class ConvNet(nn.Module):
    """Small ConvNet for CONIC."""

    def __init__(self):

        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # we use the maxpool multiple times, but define it once
        self.pool = nn.MaxPool2d(2, 2)
        # in_channels = 6 because self.conv1 output 6 channel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 5*5 comes from the dimension of the last convnet layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # no activation on final layer
        return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)




# The ``@ray.remote`` decorator defines a remote process. It wraps the
# ParameterServer class and allows users to instantiate it as a
# remote actor.


@ray.remote
class ParameterServer(object):
    def __init__(self, lr):
        self.model = ConvNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0)
            for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()


#The worker will synchronize its model with the
# Parameter Server model weights.

@ray.remote
class Worker(object):

    def __init__(self, lr=0.01, momentum=0.5):
        self.model = ConvNet()
        self.data_iterator = iter(get_data_loader()[0])
        self.time_step = 0
        self.iteration_runtime = []
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=lr, momentum=momentum)

    def compute_gradients(self, weights):
        start_time = time.time()
        self.time_step += 1
        self.model.set_weights(weights)
        try:
            data, target = next(self.data_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(get_data_loader()[0])
            data, target = next(self.data_iterator)
        self.optimizer.zero_grad()
        output = self.model(data)
        #loss = F.nll_loss(output, target)
        loss = self.criterion(output, target)
        loss.backward()
        #self.optimizer.step()
        #self.iteration_runtime = time.time() - start_time
        return self.model.get_gradients()

    def get_time_step(self):
        return self.time_step

    def set_iteration_time_current(self, time_):
        self.iteration_runtime.append(time_)

    def get_iteration_time(self, iter_count):
        return self.iteration_runtime[iter_count]