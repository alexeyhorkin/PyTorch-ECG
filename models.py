import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=4),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Conv1d(32,64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Conv1d(64,128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU()
        )
        self.l4 = nn.Sequential(
            nn.Conv1d(128,64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU()
        )
        self.l5 = nn.Sequential(
            nn.Conv1d(64,64,kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU()
        )
        self.linear_part = nn.Sequential(
            nn.Linear(64*78, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 196),
        )

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = out.view(out.size(0),-1)
        out = self.linear_part(out)
        return torch.sigmoid(out)

    
class CordConv1d(torch.nn.Module):
    def __init__(self, dim, *args, **kwargs):
        super(CordConv1d, self).__init__()
        self.dim = dim
        args = list(args)
        args[0]+=1
        args = tuple(args)
        self.conv = nn.Conv1d(*args, **kwargs)
    def forward(self, x):
        batch_size = x.shape[0]
        x_range = torch.arange(0,self.dim).repeat(batch_size,1)
        x_range = x_range[:,None,:]
        x_channel = x_range.float()/(self.dim-1)
        x_channel = x_channel*2-1
        out = torch.cat((x, x_channel),1)
        out = self.conv(out)
        return out