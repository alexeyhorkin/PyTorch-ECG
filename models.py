import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_cord(nn.Module):
    def __init__(self, **kwargs):
        super(CNN_cord, self).__init__()
        if  'output_size' not in kwargs.keys():
            kwargs['output_size'] = 1
        if  'input_size' not in kwargs.keys():
            kwargs['input_size'] = 300


        self.BN = nn.BatchNorm1d(1)
        self.layer1 = nn.Sequential(
            CoordConv1d(1, 8, kernel_size=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(8, 1, kernel_size=3, padding=1)
        )

        self.classifier = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):
        out = self.BN(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.classifier(out)
        out = out.view(out.shape[0],-1)
        return out


class CNN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()
        if  'output_size' not in kwargs.keys():
            kwargs['output_size'] = 1
        if  'input_size' not in kwargs.keys():
            kwargs['input_size'] = 300


        self.BN = nn.BatchNorm1d(1)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(8, 1, kernel_size=3, padding=1)
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(8, 1, kernel_size=3, padding=1)
        )             
        self.classifier = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):
        out = self.BN(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.classifier(out)
        out = out.view(out.shape[0],-1)
        return out



class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False, use_cuda=True):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.use_cuda = use_cuda

    def forward(self, input_tensor):
        if self.rank == 1:
            batch_size_shape, _, dim_x = input_tensor.shape
            xx_range = torch.arange(dim_x, dtype=torch.int32)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
            out = torch.cat([input_tensor, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

    
        return out


class CoordConv1d(torch.nn.modules.conv.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 1
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv1d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out

# last models arch
# class CNN(nn.Module):
#     def __init__(self, **kwargs):
#         super(CNN, self).__init__()
#         if  'output_size' not in kwargs.keys():
#             kwargs['output_size'] = 1
#         if  'input_size' not in kwargs.keys():
#             kwargs['input_size'] = 300


#         self.BN = nn.BatchNorm1d(1)
#         self.layer1 = nn.Sequential(
#             CordConv1d(kwargs['input_size'], 1, 8, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm1d(8),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2)
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv1d(8, 16, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2)
#         )
#         self.layer3 = nn.Sequential(
#             nn.Conv1d(16, 32, kernel_size=30, padding=0, bias=False),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2)
#         )
#         self.layer4 = nn.Sequential(
#             nn.Conv1d(32, 64 , kernel_size=10, padding=0, bias=False),
#             nn.ReLU()
#         )
#         self.layer5 = nn.Sequential(
#             nn.Conv1d(64, 1 , kernel_size=3, padding=0, bias=False),
#             nn.ReLU()
#         )
#         self.classifier = nn.AdaptiveAvgPool1d(output_size=1)

#     def forward(self, x):
#         out = self.BN(x)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.layer5(out)
#         out = self.classifier(out)
#         out = out.view(out.shape[0],-1)
#         return out

    
# class CoordConv1d(torch.nn.Module):
#     def __init__(self, dim, *args, **kwargs):
#         super(CoordConv1d, self).__init__()
#         self.dim = dim
#         self.rank = 1
#         self.register_buffer('x_range', torch.arange(0,self.dim))
#         self.conv = nn.Conv1d(args[0] + self.rank, *args[1:],**kwargs)
#     def forward(self, x):
#         batch_size = x.shape[0]
#         x_range = self.x_range.repeat(batch_size,1)
#         x_range = x_range[:,None,:]
#         x_channel = x_range.float()/(self.dim-1)
#         x_channel = x_channel*2-1
#         out = torch.cat((x, x_channel),1)
#         out = self.conv(out)
#         return out