import torch
import numpy as np

#定义自己的网络模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mlp1 = torch.nn.Linear(1024, 512)
        self.mpl2 = torch.nn.Linear(512, 256)
        self.mlp3 = torch.nn.Linear(256, 40)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = x.unsqueeze(-1) 
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x,_ = torch.max(x,dim=0,keepdim=True)
        x = x.squeeze(dim=-1).squeeze(dim=0)
        x = self.relu(self.mlp1(x))
        x = self.relu(self.mpl2(x))
        x = self.mlp3(x).unsqueeze(dim = 0)
        return x

