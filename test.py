import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms, models
from collections import Counter
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


from model import MyModel
from load_dataset import create_test_loader

def accuracy(model, loader):
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
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('classification_report.csv')

    cf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(cf_matrix)
    plt.show()
    
    acc = n_correct / n_total
    return acc

batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dir = './data/test/'
test_loader = create_test_loader(test_dir, batch_size)

model = MyModel(38)
model.to(device)

model.load_state_dict(torch.load("./saved_models/model_0.15888998160303078_16.pt", map_location=torch.device('cpu')))
model.eval()

test_acc = accuracy(model, test_loader)
print(
    f"Test Accuracy : {test_acc}"
)