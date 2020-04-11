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
            nn.Linear(kwargs['input_size'], 300),
            nn.BatchNorm1d(300),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(300,100),
            nn.BatchNorm1d(100),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(100,20),
            nn.BatchNorm1d(20),
            nn.ReLU()
        )
        self.output = nn.Linear(20, kwargs['output_size'])
    
    def forward(self, x):
        out = x.view(x.shape[0],-1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return self.output(out)
        

    
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