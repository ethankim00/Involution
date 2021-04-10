import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from datasets import load_cifar
from resnet import ResNet18
from involution import Involution2d
from torch.nn import ReLU

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy



class CNN(pl.LightningModule):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(CNN, self).__init__()
        self.accuracy = pl.metrics.Accuracy()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_acc_step', self.accuracy(y_hat, y), prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        return val_loss
        

class CifarCNN(pl.LightningModule):
    pass

class CifarInvNet(pl.LightningModule):

    def __init__(self):
        super(CifarInvNet, self).__init__()
        self.accuracy = pl.metrics.Accuracy()
        activation = ReLU()
        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            Involution2d(in_dim=3, out_dim=32, kernel_dim = 3, stride = 1, groups=1, reduction=1, dilation  = 1, padding = 1, activation=activation),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            Involution2d(in_dim=32, out_dim=64, kernel_dim = 3, stride = 1, groups=1, reduction=1, dilation=1, padding = 1, activation=activation),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            Involution2d(in_dim=64, out_dim=128, kernel_dim = 3, stride = 1, groups=1, reduction=1, dilation=1, padding=1, activation=activation),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Involution2d(in_dim=128, out_dim=128, kernel_dim=3, stride=1, groups=1, reduction=1, dilation=1, padding=1, activation=activation),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            Involution2d(in_dim=128, out_dim=128, kernel_dim = 3, stride = 1, groups=1, reduction=1, dilation  = 1, padding = 1, activation=activation),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Involution2d(in_dim=128, out_dim=128, kernel_dim = 3, stride = 1, groups=1, reduction=1, dilation  = 1, padding = 1, activation=activation),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_acc_step', self.accuracy(y_hat, y), prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        return val_loss

def run():
    model = ResNet18('inv', 10)
    train, val = load_cifar(128)

    # Initialize a trainer
    trainer = pl.Trainer(gpus = 1, max_epochs=15, progress_bar_refresh_rate=20)

    # Train the model 
    trainer.fit(model, train, val)
    trainer.test(test_dataloaders=val)
    trainer.save_checkpoint("cifarconv.ckpt")
if __name__ == '__main__':
    run()
