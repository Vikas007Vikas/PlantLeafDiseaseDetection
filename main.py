import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms, models
from collections import Counter
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import classification_report
import torch.nn as nn
import torch.optim as optim
# from torchsummary import summary
from datetime import datetime
from train import batch_gd
from model import MyModel
from load_dataset import create_torch_loaders, create_test_loader

def accuracy(loader):
    n_correct = 0
    n_total = 0
    y_pred = []
    y_true = []
    softmax = nn.Softmax(dim=1)

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        predictions = torch.argmax(softmax(outputs),dim=1)
        y_pred.extend(predictions)
        y_true.extend(targets)

        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    y_true = torch.tensor(y_true, device = 'cpu')
    y_pred = torch.tensor(y_pred, device = 'cpu')
    print(classification_report(y_true,y_pred))
    
    acc = n_correct / n_total
    return acc

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, required=True, help="Please specify 1 if trained and 0 if not.")
args = parser.parse_args()

batch_size = 32
data_dir = '../../data/train'
dataset, train_loader, validation_loader = create_torch_loaders(data_dir, batch_size)

targets_size = len(dataset.class_to_idx)

# run on GPU if available else run on a CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MyModel(targets_size)
model.to(device)

#print(summary(model, (3, 224, 224)))

criterion = nn.CrossEntropyLoss()  # this include softmax + cross entropy loss
optimizer = torch.optim.Adam(model.parameters())

if args.mode == 0:
    model, train_losses, validation_losses = batch_gd(
        model, optimizer, criterion, train_loader, validation_loader, 100, device
    )
    torch.save(model.state_dict() , 'dm_model_main.pt')

else:
    # Model already trained. So, load it from the saved model file
    model.load_state_dict(torch.load("dm_model_main.pt"))
    model.eval()

train_acc = accuracy(train_loader)
validation_acc = accuracy(validation_loader)

test_dir = '../../data/test/'
test_loader = create_test_loader(test_dir, batch_size)
test_acc = accuracy(test_loader)

print(
    f"Train Accuracy : {train_acc}\nTest Accuracy : {test_acc}\nValidation Accuracy : {validation_acc}"
)



