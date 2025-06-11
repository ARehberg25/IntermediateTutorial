"""
Simple CNN based on Kazu Terao's and Abhishek Abhishek code .
"""

# PyTorch imports
import torch.nn as nn

import numpy as np

# KazuNet class
class SimpleCNN(nn.Module):
    
    # Initializer
    
    def __init__(self, config, num_input_channels=38, num_classes=3, train=True):
        
        # Initialize the superclass
        super(SimpleCNN, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        
        # Activation functions
        self.relu = nn.ReLU()
                
        # Feature extractor

        self.f_embed = nn.Conv2d(num_input_channels, 2, kernel_size=1, stride=1, padding=0)
        # Convolutions and max-pooling
        self.f_conv1 = nn.Conv2d(2, 4, kernel_size=config.kernel_size, stride=1, padding=1)
        self.f_max_pool1  = nn.MaxPool2d(2,2)
        
        self.f_conv2a = nn.Conv2d(4, 4, kernel_size=config.kernel_size, stride=1, padding=1)
        self.f_conv2b = nn.Conv2d(4, 4, kernel_size=config.kernel_size, stride=1, padding=1)
        self.f_max_pool2  = nn.MaxPool2d(2,2)
        
        self.f_conv3a = nn.Conv2d(4, 8, kernel_size=config.kernel_size, stride=1, padding=1)
        self.f_conv3b = nn.Conv2d(8, 8, kernel_size=config.kernel_size, stride=1, padding=1)
        self.f_max_pool3 = nn.MaxPool2d(2,2)
        
        self.f_conv4  = nn.Conv2d(8, 8, kernel_size=2, stride=1, padding=0)
        
        # Flattening / MLP
        
        # Fully-connected layers
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, num_classes)
        
    # Forward pass
    
    def forward(self, x):
        
        # Convolutions and max-pooling
        x = self.f_max_pool1(self.relu(self.f_conv1(self.relu(self.f_embed(x)))))
        x = self.f_max_pool2(self.relu(self.f_conv2b(self.relu(self.f_conv2a(x)))))
        x = self.f_max_pool3(self.relu(self.f_conv3b(self.relu(self.f_conv3a(x)))))
        x = self.relu(self.f_conv4(x))

        
        # Flattening
        x = nn.MaxPool2d(x.size()[2:])(x)
        x = x.view(-1, 8)
        
        # Fully-connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
