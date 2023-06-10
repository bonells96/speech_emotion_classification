import torch
import torch.nn as nn
from xgboost import XGBClassifier
from torch.nn import functional as F


################################################ Neural network ################################################

class StandardNet(nn.Module):
    def __init__(self, input_size, hidden_size=100):

        self.softmax = nn.Softmax(dim=1)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 7)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.softmax(self.fc3(x))
        return out
    
        
################################################ Neural network ################################################

class StandardNetBD(nn.Module):
    def __init__(self, input_size, hidden_size=100):

        self.softmax = nn.Softmax(dim=1)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 7)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.softmax(self.fc3(x))
        return out