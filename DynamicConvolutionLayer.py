import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F

class DCM(nn.Module):
    def __init__(self, in_channel, out_channel, filter_size=3):
        super(DCM, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.filter_size = filter_size
        self.norm = nn.Sequential(nn.BatchNorm2d(out_channel, affine=True),
                                  nn.ReLU(),
                                  )

    def forward(self, x):
        pre_filter = F.adaptive_avg_pool2d(x, 3)
        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        pre_filter = pre_filter.view(b * c, 1, self.filter_size, self.filter_size)
        # padding for input features
        pad = (self.filter_size - 1) // 2
        if (self.filter_size - 1) % 2 == 0:
            p2d = (pad, pad, pad, pad)
        else:
            p2d = (pad + 1, pad, pad + 1, pad)
        x = F.pad(input=x, pad=p2d, mode='constant', value=0)
        output = F.conv2d(input=x, weight=pre_filter, groups=b * c)  # 实现的是depth-wise convolution
        # 若要实现普通卷积，则需将预测卷积核pre_filter凑成 (self.out_channel, self,in_channel, k, k)
        # 简单使用F.adaptive_avg_pool2d()可能凑不出来，需要用卷积生成去处理batch_size
        output = output.view(b, c, h, w)
        output = self.norm(output)
        return output

#
# if __name__ == '__main__':
#     dcm = DCM()
#     x = Variable(torch.rand(3, 3, 2, 2))
#     y = dcm(x)
#     print(x)
#     print('#####################')
#     print(y)
#     print('##################')
#     conv2d = nn.Conv2d(3, 2, 1)
#     z = conv2d(x)
#     print(z)