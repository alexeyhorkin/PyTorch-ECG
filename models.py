import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__()
        if  'input_size' not in kwargs.keys():
            kwargs['input_size'] = 512
        if  'output_size' not in kwargs.keys():
            kwargs['output_size'] = 1

        self.layer1 = nn.Sequential(
            nn.Linear(kwargs['input_size'], 70),
            nn.BatchNorm1d(70),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(70,20),
            nn.BatchNorm1d(20),
            nn.ReLU()
        )
        self.output = nn.Linear(20, kwargs['output_size'])
    
    def forward(self, x):
        out = x.view(x.shape[0],-1)
        out = self.layer1(out)
        out = self.layer2(out)
        return self.output(out)



class CNN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()
        if  'output_size' not in kwargs.keys():
            kwargs['output_size'] = 1
        if  'input_size' not in kwargs.keys():
            kwargs['input_size'] = 300


        self.BN = nn.BatchNorm1d(1)
        self.layer1 = nn.Sequential(
            CordConv1d(kwargs['input_size'], 1, 8, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=30, padding=0, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(32, 64 , kernel_size=10, padding=0, bias=False),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 1 , kernel_size=3, padding=0, bias=False),
            nn.ReLU()
        )
        self.classifier = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        out = self.BN(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.classifier(out)
        out = out.view(out.shape[0],-1)
        return out

    
class CordConv1d(torch.nn.Module):
    def __init__(self, dim, *args, **kwargs):
        super(CordConv1d, self).__init__()
        self.dim = dim
        args = list(args)
        args[0]+=1
        args = tuple(args)
        self.register_buffer('x_range', torch.arange(0,self.dim))
        self.conv = nn.Conv1d(*args, **kwargs)
    def forward(self, x):
        batch_size = x.shape[0]
        x_range = self.x_range.repeat(batch_size,1)
        x_range = x_range[:,None,:]
        x_channel = x_range.float()/(self.dim-1)
        x_channel = x_channel*2-1
        out = torch.cat((x, x_channel),1)
        out = self.conv(out)
        return out
        