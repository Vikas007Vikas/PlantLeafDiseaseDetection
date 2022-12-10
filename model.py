import torch
import torch.nn as nn
import torchvision
from torchvision import models

class ResNet101(nn.Module):
    def __init__(self, K):
        super(ResNet101, self).__init__()
        self.pretrained_model = torchvision.models.resnet101(pretrained=True)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        
        num_ftrs = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(num_ftrs, K)

    def forward(self, X):
        out = self.pretrained_model(X)
        return out

class AlexNet(nn.Module):
    def __init__(self, K):
        super(AlexNet, self).__init__()
        self.pretrained_model = torchvision.models.alexnet(pretrained=True)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        
        self.pretrained_model.classifier[4] = nn.Linear(4096,1024)
        self.pretrained_model.classifier[6] = nn.Linear(1024,K)

    def forward(self, X):
        out = self.pretrained_model(X)
        return out

class MyModel(nn.Module):
    def __init__(self, K):

        super(MyModel, self).__init__()

        self.autoencoder = AutoEncoder()
        self.ResNet = ResNet101(K)

    def forward(self, x):
        reconstructed, out_encoder = self.autoencoder(x)
        out = self.ResNet(out_encoder)
        return out, reconstructed


class AutoEncoder(nn.Module):
    def __init__(self):

        super(AutoEncoder, self).__init__()

        self.encoder = Encoder()
        self.unpoolLayer = nn.MaxUnpool2d(10, 2)
        self.decoder = Decoder()
    
    def forward(self, x):

        out, indices = self.encoder(x)
        x = self.unpoolLayer(out, indices)
        x = self.decoder(x)

        return x, out

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=1, padding=1),#460
            nn.ReLU(True),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 5, stride=1, padding=0), #229
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 3, 3, stride=1, padding=1), #225
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.MaxPool2d(10, 2, return_indices=True)
        )
    
    def forward(self, x):
        x = self.encoder_cnn(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(3, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(3)
        )
    
    def forward(self, x):
        x = self.decoder_cnn(x)
        return x