"""
Simple MLP.
"""

# PyTorch imports
import torch.nn as nn

import numpy as np

# 4 hidden layer MLP
class SimpleMLP(nn.Module):
    
    #Define the network layers here
    def __init__(self, num_classes=3):
        
        # Initialize the superclass
        super(SimpleMLP, self).__init__()
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        
        # Fully-connected layers
        self.fc1 = nn.Linear(24320, 160)
        self.fc2 = nn.Linear(160, 40)
        self.fc3 = nn.Linear(40, num_classes)
        
    # Forward pass
    
    def forward(self, x):
        
        #Put into the right shape
        x=x.view(-1, np.prod(x.size()[1:]))
        #print(x.shape)
        # Fully-connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

#Sequential model: not used in TRISEP tutorial
class SimpleMLPSEQ(nn.Module):
    
    #Define the network layers - using a Sequential Module
    def __init__(self,num_classes=3):
        
        # Initialize the superclass
        super(SimpleMLPSEQ, self).__init__()

        self._sequence = nn.Sequential(
            nn.Linear(24320, 972),nn.ReLU(),
            nn.Linear(972, 486),nn.ReLU(),      
            nn.Linear(484, 160),nn.ReLU(),      
            nn.Linear(160, 40),nn.ReLU(),       
            nn.Linear(40, num_classes))
        
        
    # Forward pass
    
    def forward(self, x):
        #Put into the right shape
        x=x.view(x.shape[0],-1)
        x=self._sequence(x)
        
        return x
