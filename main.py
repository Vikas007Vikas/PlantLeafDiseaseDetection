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
from torchsummary import summary
from datetime import datetime
from train import batch_gd
from model import ResNet101, AlexNet, MyModel
from load_dataset import create_torch_loaders, create_test_loader

def accuracy(loader, modelType):
    n_correct = 0
    n_total = 0
    y_pred = []
    y_true = []
    softmax = nn.Softmax(dim=1)

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        if (modelType == 2):
            output_model, out_autoencoder = model(inputs)
        else:
            output_model = model(inputs)

        predictions = torch.argmax(softmax(output_model),dim=1)
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
parser.add_argument('--model', type=int, required=True, help="Please specify 0 if ResNet101 and 1 if AlexNet.")
args = parser.parse_args()

batch_size = 32
data_dir = './segmented_data_60_40/train'
dataset, train_loader, validation_loader = create_torch_loaders(data_dir, batch_size)

targets_size = len(dataset.class_to_idx)

# run on GPU if available else run on a CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if (args.model == 0):
    model = ResNet101(targets_size)
elif (args.model == 1):
    model = AlexNet(targets_size)
else:
    model = MyModel(targets_size)
model.to(device)

summary(model, (3, 460, 460))
class_wts = [2.18792477, 2.38097695, 4.95631937, 0.87359586, 0.94010187, 1.38776942
            , 1.68263036, 2.83493754, 1.22450243, 1.46300993, 1.25185386, 1.20227549
            , 1.02906631, 1.3319542, 3.35750667, 0.2614205, 0.62754431, 4.02529805
            , 1.4147164, 0.98059078, 1.42858617, 1.42023187, 9.5865651, 3.91709111
            , 0.28033049, 0.76451096, 1.30569704, 2.96170304, 0.67586173, 1.4147164
            , 0.73742808, 1.46595362, 0.80505961, 0.85313694, 1.02472426, 0.26455299
            , 3.95966819, 0.90059202]
class_wts = torch.FloatTensor(class_wts)
criterion1 = nn.CrossEntropyLoss(weight=class_wts)  # this include softmax + cross entropy loss
criterion2 = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

if args.mode == 0:
    model, train_losses, validation_losses = batch_gd(
        model, optimizer, criterion1, criterion2, train_loader, validation_loader, 1, device, args.model
    )
    torch.save(model.state_dict() , 'dm_model_main.pt')

else:
    # Model already trained. So, load it from the saved model file
    model.load_state_dict(torch.load("dm_model_main.pt"))
    model.eval()

train_acc = accuracy(train_loader, args.model)
validation_acc = accuracy(validation_loader, args.model)

test_dir = './segmented_data_60_40/test/'
test_loader = create_test_loader(test_dir, batch_size, args.model)
test_acc = accuracy(test_loader)

print(
    f"Train Accuracy : {train_acc}\nTest Accuracy : {test_acc}\nValidation Accuracy : {validation_acc}"
)



