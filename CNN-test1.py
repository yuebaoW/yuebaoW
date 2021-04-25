import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

print("PyTorch Version:" ,torch.__version__)

minist_data = datasets.MNIST("./mnist_data", train=True,download=True)

print(minist_data[5][0])