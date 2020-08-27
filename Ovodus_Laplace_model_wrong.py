"""
Copyright (c) 2019 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import torch
import torch.nn as nn
import numpy as np
from DynamicConvolutionLayer import DCM



def imgGaussian(sigma):
    '''
    :param sigma: σstandard deviation
    :return: Gaussian filter template
    '''
    img_h = img_w = 2 * sigma + 1
    gaussian_mat = np.zeros((img_h, img_w))
    for x in range(-sigma, sigma + 1):
        for y in range(-sigma, sigma + 1):
            gaussian_mat[x + sigma][y + sigma] = np.exp(-0.5 * (x ** 2 + y ** 2) / (sigma ** 2))
    return gaussian_mat / gaussian_mat.sum()

def get_lap_sob_gaus(in_channels, pool=True):
    """Laplace decomposition using conv2d"""
    Lap_ope_E = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])
    Lap_ope_D = np.array([[1, 1, 1],
                          [1, -8, 1],
                          [1, 1, 1]])
    Gauss_mat = imgGaussian(1)
    Sobel_ope_V = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    Sobel_ope_H = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    filter_Gaus = torch.from_numpy(Gauss_mat).unsqueeze(0)
    filter_Lap_E = torch.from_numpy(Lap_ope_E).unsqueeze(0)
    filter_Lap_D = torch.from_numpy(Lap_ope_D).unsqueeze(0)
    filter_Sob_V = torch.from_numpy(Sobel_ope_V).unsqueeze(0)
    filter_Sob_H = torch.from_numpy(Sobel_ope_H).unsqueeze(0)


    if pool:
        layer = nn.Conv2d
        Gauss_mat_pool = layer(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False,
                               groups=in_channels)
        Lap_ope_E_pool = layer(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False,
                               groups=in_channels)
        Lap_ope_D_pool = layer(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False,
                               groups=in_channels)
        Sobel_ope_V_pool = layer(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False,
                                 groups=in_channels)
        Sobel_ope_H_pool = layer(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False,
                                 groups=in_channels)
    else:
        layer = nn.ConvTranspose2d
        Gauss_mat_pool = layer(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, groups=in_channels)
        Lap_ope_E_pool = layer(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, groups=in_channels)
        Lap_ope_D_pool = layer(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, groups=in_channels)
        Sobel_ope_V_pool = layer(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, groups=in_channels)
        Sobel_ope_H_pool = layer(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, groups=in_channels)

    Gauss_mat_pool.weight.requires_grad = False
    Lap_ope_E_pool.weight.requires_grad = False
    Lap_ope_D_pool.weight.requires_grad = False
    Sobel_ope_V_pool.weight.requires_grad = False
    Sobel_ope_H_pool.weight.requires_grad = False

    Gauss_mat_pool.weight.data = filter_Gaus.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    Lap_ope_E_pool.weight.data = filter_Lap_E.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    Lap_ope_D_pool.weight.data = filter_Lap_D.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    Sobel_ope_V_pool.weight.data = filter_Sob_V.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    Sobel_ope_H_pool.weight.data = filter_Sob_H.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return Gauss_mat_pool, Lap_ope_E_pool, Lap_ope_D_pool, Sobel_ope_V_pool, Sobel_ope_H_pool

class Lap_Sob_GausPool(nn.Module):
    def __init__(self, in_channels):
        super(Lap_Sob_GausPool, self).__init__()
        self.GM, self.LE, self.LD, self.SV, self.SH = get_lap_sob_gaus(in_channels)
    def forward(self, x):
        out = self.GM(x)
        return out, self.LE(out), self.LD(out), self.SV(out), self.SH(out)

class Lap_Sob_GausUnPool(nn.Module):
    def __init__(self, in_channels, option_pool='sum'):
        super(Lap_Sob_GausUnPool, self).__init__()
        self.option_pool = option_pool
        self.GM, self.LE, self.LD, self.SV, self.SH = get_lap_sob_gaus(in_channels, pool=False)

    def forward(self, GM, LE, LD, SV, SH, primary=None):
        if self.option_pool == 'sum':
            return self.GM(GM) + self.GM(self.LE(LE)) + self.GM(self.LD(LD)) + self.GM(self.SV(SV)) + self.GM(self.SH(SH))
        elif self.option_pool == 'cat6' and primary is not None:
            with torch.no_grad():
                return torch.cat([self.GM(GM), self.GM(self.LE(LE)), self.GM(self.LD(LD)), self.GM(self.SV(SV)),  self.GM(self.SH(SH)), primary], dim=1)
        else:
            raise NotImplementedError

class Lap_Sob_GausEncoder(nn.Module):
    def __init__(self, option_pool):
        super(Lap_Sob_GausEncoder, self).__init__()
        self.option_pool = option_pool

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv0_1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.pool1 = Lap_Sob_GausPool(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv2_3 = nn.Conv2d(128, 128, 3, 1, 0)
        self.pool2 = Lap_Sob_GausPool(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.pool3 = Lap_Sob_GausPool(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)
        self.conv4_2 = nn.Conv2d(256, 512, 3, 1, 0)

        self.dcm_64 = DCM(64, 64)
        self.dcm_128 = DCM(128, 128)
        self.dcm_256 = DCM(256, 256)
        self.dcm_512 = DCM(512, 512)

    def forwark(self, x):
        skips = {}
        for level in {1, 2, 3, 4}:
            x = self.encode(x, skips, level)
        return x

    def encode(self, x, skips, level):
        assert  level in {1, 2, 3, 4}
        if self.option_pool == 'sum':
            if level == 1:
                out = self.conv0(x)      # 3 -> 3
                out = self.relu(self.conv1_1(self.pad(out)))  # 3 -> 64
                residual = out
                #out = self.dcm_64(out)                        # 64 -> 64
                out = self.relu(self.conv1_2(self.pad(out)))  # 64 - > 64
                skips['conv1_2'] = out
                out = out + residual
                GM, LE, LD, SV, SH = self.pool1(out)
                skips['pool1'] = [LE, LD, SV, SH]
                return GM
            elif level == 2:
                x = self.relu(self.conv2_1(self.pad(x)))  # 64 -> 128
                residual = x
                out = self.relu(self.conv2_2(self.pad(x)))    # 128 -> 128
                out = self.dcm_128(out)                       # 128 -> 128
                out = self.relu(self.conv2_3(self.pad(x)))  # 128 -> 128
                skips['conv2_3'] = out
                out = out + residual
                GM, LE, LD, SV, SH = self.pool2(out)
                skips['pool2'] = [LE, LD, SV, SH]
                return GM
            elif level == 3:
                x = self.relu(self.conv3_1(self.pad(x)))    # 128 -> 256
                residual = x
                out = self.relu(self.conv3_2(self.pad(x)))  # 256 -> 256
                out = self.relu(self.conv3_3(self.pad(out)))  # 256 -> 256
                out = self.dcm_256(out)
                out = self.relu(self.conv3_4(self.pad(x)))  # 256 -> 256
                out = out + residual
                skips['conv3_4'] = out
                GM, LE, LD, SV, SH = self.pool3(out)
                skips['pool3'] = [LE, LD, SV, SH]
                return GM
            else:
                out = self.relu(self.conv4_1(self.pad(x)))    # 256 -> 512
                residual = out
                out = self.dcm_512(out)                       # 512 -> 512
                out = self.relu(self.conv4_2(self.pad(x)))             # 512 -> 512
                out = out + residual
                return out

        elif self.option_pool == 'cat6':
            if level == 1:
                out = self.conv0(x)           # 3 -> 3
                out = self.relu(self.conv1_1(self.pad(out)))     # 3 -> 64
                out = self.dcm_64(out)  # 64 -> 64
                return out

            elif level == 2:
                out = self.relu(self.conv1_2(self.pad(x)))      # 64 -> 64
                skips['conv1_2'] = out
                GM, LE, LD, SV, SH = self.pool1(out)
                skips['pool1'] = [LE, LD, SV, SH]
                out = self.relu(self.conv2_1(self.pad(GM)))     # 64 -> 128
                out = self.dcm_128(out)
                return out

            elif level == 3:
                out = self.relu(self.conv2_2(self.pad(x)))      # 128 -> 128
                skips['conv2_2'] = out
                GM, LE, LD, SV, SH = self.pool2(out)
                skips['pool2'] = [LE, LD, SV, SH]
                out = self.relu(self.conv3_1(self.pad(GM)))    #128 -> 256
                out = self.dcm_256(out)                       #256 -> 256
                return out

            else:
                out = self.relu(self.conv3_2(self.pad(x)))      #256 -> 256
                out = self.relu(self.conv3_3(self.pad(out)))    #256 -> 256
                out = self.dcm_256(out)                         #256 -> 256
                out = self.relu(self.conv3_4(self.pad(out)))     #256 -> 256
                skips['conv3_4'] = out
                GM, LE, LD, SV, SH = self.pool3(out)
                skips['pool3'] = [LE, LD, SV, SH]
                out = self.relu(self.conv4_1(self.pad(GM)))      # 256 -> 512
                out = self.dcm_512(out)  # 512 -> 512
                return out
        else:
            raise NotImplementedError

class Lap_Sob_GausDecoder(nn.Module):
    def __init__(self, option_pool):
        super(Lap_Sob_GausDecoder, self).__init__()
        self.option_pool = option_pool

        if option_pool == 'sum':
            multiply_in = 1
        elif option_pool == 'cat6':
            multiply_in = 6
        else:
            raise NotImplementedError

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)

        self.dcm256 = DCM(256, 256)
        self.recon_block3 = Lap_Sob_GausUnPool(256, option_pool)
        if option_pool == 'sum':
            self.conv3_4 = nn.Conv2d(256*multiply_in, 256, 3, 1, 0)
        else:
            self.conv3_4_2 = nn.Conv2d(256*multiply_in, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)
        self.dcm128 = DCM(128, 128)


        self.recon_block2 = Lap_Sob_GausUnPool(128, option_pool)
        if option_pool == 'sum':
            self.conv2_2 = nn.Conv2d(128*multiply_in, 128, 3, 1, 0)
        else:
            self.conv2_2_2 = nn.Conv2d(128*multiply_in, 128, 3, 1, 0)
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)
        self.dcm64 = DCM(64, 64)

        self.recon_block1 = Lap_Sob_GausUnPool(64, option_pool)
        if option_pool == 'sum':
            self.conv1_2 = nn.Conv2d(64*multiply_in, 64, 3, 1, 0)
        else:
            self.conv1_2_2 = nn.Conv2d(64*multiply_in, 64, 3, 1, 0)
        self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)
        self.dcm3 = DCM(3, 3)

    def forward(self, x, skips):
        for level in [4, 3, 2, 1]:
            x = self.decode(x, skips, level)
        return x

    def decode(self, x, skips, level):
        assert level in {4, 3, 2, 1}
        if level == 4:
            out = self.relu(self.conv4_1(self.pad(x)))
            #out = self.dcm256(out)
            LE, LD, SV, SH = skips['pool3']
            primary = skips['conv3_4'] if 'conv3_4' in skips.keys() else None
            out = self.recon_block3(out, LE, LD, SV, SH, primary)
            _conv3_4 = self.conv3_4 if self.option_pool == 'sum' else self.conv3_4_2
            out = self.relu(_conv3_4(self.pad(out)))
            out = self.relu(self.conv3_3(self.pad(out)))
            out = self.relu(self.conv3_2(self.pad(out)))
            return out
        elif level == 3:
            out = self.relu(self.conv3_1(self.pad(x)))
            #out = self.pad(0.5)
            #out = self.dcm128(out)
            LE, LD, SV, SH = skips['pool2']
            primary = skips['conv2_2'] if 'conv2_2' in skips.keys() else None
            out = self.recon_block2(out, LE, LD, SV, SH, primary)
            _conv2_2 = self.conv2_2 if self.option_pool == 'sum' else self.conv2_2_2
            return self.relu(_conv2_2(self.pad(out)))
        elif level == 2:
            out = self.relu(self.conv2_1(self.pad(x)))
            #self.dcm64(out)
            LE, LD, SV, SH = skips['pool1']
            primary = skips['conv1_2'] if 'conv1_2' in skips.keys() else None
            out = self.recon_block1(out, LE, LD, SV, SH, primary)
            _conv1_2 = self.conv1_2 if self.option_pool == 'sum' else self.conv1_2_2
            return self.relu(_conv1_2(self.pad(out)))
        else:
            out = self.conv1_1(self.pad(x))
            #out = self.dcm3(out)
            return out


# def get_wav(in_channels, pool=True):
#     """Laplace decomposition using conv2d"""
#
#     """wavelet decomposition using conv2d"""
#     harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
#     harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
#     harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]
#
#     harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
#     harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
#     harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
#     harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H
#
#     filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
#     filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
#     filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
#     filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)
#
#     if pool:
#         net = nn.Conv2d
#     else:
#         net = nn.ConvTranspose2d
#
#     LL = net(in_channels, in_channels,
#              kernel_size=2, stride=2, padding=0, bias=False,
#              groups=in_channels)
#     LH = net(in_channels, in_channels,
#              kernel_size=2, stride=2, padding=0, bias=False,
#              groups=in_channels)
#     HL = net(in_channels, in_channels,
#              kernel_size=2, stride=2, padding=0, bias=False,
#              groups=in_channels)
#     HH = net(in_channels, in_channels,
#              kernel_size=2, stride=2, padding=0, bias=False,
#              groups=in_channels)
#
#     LL.weight.requires_grad = False
#     LH.weight.requires_grad = False
#     HL.weight.requires_grad = False
#     HH.weight.requires_grad = False
#
#     LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
#     LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
#     HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
#     HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
#
#     return LL, LH, HL, HH
#
#
# class WavePool(nn.Module):
#     def __init__(self, in_channels):
#         super(WavePool, self).__init__()
#         self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)
#
#     def forward(self, x):
#         return self.LL(x), self.LH(x), self.HL(x), self.HH(x)
#
#
# class WaveUnpool(nn.Module):
#     def __init__(self, in_channels, option_unpool='cat5'):
#         super(WaveUnpool, self).__init__()
#         self.in_channels = in_channels
#         self.option_unpool = option_unpool
#         self.LL, self.LH, self.HL, self.HH = get_wav(self.in_channels, pool=False)
#
#     def forward(self, LL, LH, HL, HH, original=None):
#         if self.option_unpool == 'sum':
#             return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
#         elif self.option_unpool == 'cat5' and original is not None:
#             return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
#         else:
#             raise NotImplementedError
#
#
# class WaveEncoder(nn.Module):
#     def __init__(self, option_unpool):
#         super(WaveEncoder, self).__init__()
#         self.option_unpool = option_unpool
#
#         self.pad = nn.ReflectionPad2d(1)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
#         self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
#         self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
#         self.pool1 = WavePool(64)
#
#         self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
#         self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
#         self.pool2 = WavePool(128)
#
#         self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
#         self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
#         self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
#         self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
#         self.pool3 = WavePool(256)
#
#         self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)
#
#     def forward(self, x):
#         skips = {}
#         for level in [1, 2, 3, 4]:
#             x = self.encode(x, skips, level)
#         return x
#
#     def encode(self, x, skips, level):
#         assert level in {1, 2, 3, 4}
#         if self.option_unpool == 'sum':
#             if level == 1:
#                 out = self.conv0(x)
#                 out = self.relu(self.conv1_1(self.pad(out)))
#                 out = self.relu(self.conv1_2(self.pad(out)))
#                 skips['conv1_2'] = out
#                 LL, LH, HL, HH = self.pool1(out)
#                 skips['pool1'] = [LH, HL, HH]
#                 return LL
#             elif level == 2:
#                 out = self.relu(self.conv2_1(self.pad(x)))
#                 out = self.relu(self.conv2_2(self.pad(out)))
#                 skips['conv2_2'] = out
#                 LL, LH, HL, HH = self.pool2(out)
#                 skips['pool2'] = [LH, HL, HH]
#                 return LL
#             elif level == 3:
#                 out = self.relu(self.conv3_1(self.pad(x)))
#                 out = self.relu(self.conv3_2(self.pad(out)))
#                 out = self.relu(self.conv3_3(self.pad(out)))
#                 out = self.relu(self.conv3_4(self.pad(out)))
#                 skips['conv3_4'] = out
#                 LL, LH, HL, HH = self.pool3(out)
#                 skips['pool3'] = [LH, HL, HH]
#                 return LL
#             else:
#                 return self.relu(self.conv4_1(self.pad(x)))
#
#         elif self.option_unpool == 'cat5':
#             if level == 1:
#                 out = self.conv0(x)
#                 out = self.relu(self.conv1_1(self.pad(out)))
#                 return out
#
#             elif level == 2:
#                 out = self.relu(self.conv1_2(self.pad(x)))
#                 skips['conv1_2'] = out
#                 LL, LH, HL, HH = self.pool1(out)
#                 skips['pool1'] = [LH, HL, HH]
#                 out = self.relu(self.conv2_1(self.pad(LL)))
#                 return out
#
#             elif level == 3:
#                 out = self.relu(self.conv2_2(self.pad(x)))
#                 skips['conv2_2'] = out
#                 LL, LH, HL, HH = self.pool2(out)
#                 skips['pool2'] = [LH, HL, HH]
#                 out = self.relu(self.conv3_1(self.pad(LL)))
#                 return out
#
#             else:
#                 out = self.relu(self.conv3_2(self.pad(x)))
#                 out = self.relu(self.conv3_3(self.pad(out)))
#                 out = self.relu(self.conv3_4(self.pad(out)))
#                 skips['conv3_4'] = out
#                 LL, LH, HL, HH = self.pool3(out)
#                 skips['pool3'] = [LH, HL, HH]
#                 out = self.relu(self.conv4_1(self.pad(LL)))
#                 return out
#         else:
#             raise NotImplementedError
#
#
# class WaveDecoder(nn.Module):
#     def __init__(self, option_unpool):
#         super(WaveDecoder, self).__init__()
#         self.option_unpool = option_unpool
#
#         if option_unpool == 'sum':
#             multiply_in = 1
#         elif option_unpool == 'cat5':
#             multiply_in = 5
#         else:
#             raise NotImplementedError
#
#         self.pad = nn.ReflectionPad2d(1)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)
#
#         self.recon_block3 = WaveUnpool(256, option_unpool)
#         if option_unpool == 'sum':
#             self.conv3_4 = nn.Conv2d(256*multiply_in, 256, 3, 1, 0)
#         else:
#             self.conv3_4_2 = nn.Conv2d(256*multiply_in, 256, 3, 1, 0)
#         self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
#         self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
#         self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)
#
#         self.recon_block2 = WaveUnpool(128, option_unpool)
#         if option_unpool == 'sum':
#             self.conv2_2 = nn.Conv2d(128*multiply_in, 128, 3, 1, 0)
#         else:
#             self.conv2_2_2 = nn.Conv2d(128*multiply_in, 128, 3, 1, 0)
#         self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)
#
#         self.recon_block1 = WaveUnpool(64, option_unpool)
#         if option_unpool == 'sum':
#             self.conv1_2 = nn.Conv2d(64*multiply_in, 64, 3, 1, 0)
#         else:
#             self.conv1_2_2 = nn.Conv2d(64*multiply_in, 64, 3, 1, 0)
#         self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)
#
#     def forward(self, x, skips):
#         for level in [4, 3, 2, 1]:
#             x = self.decode(x, skips, level)
#         return x
#
#     def decode(self, x, skips, level):
#         assert level in {4, 3, 2, 1}
#         if level == 4:
#             out = self.relu(self.conv4_1(self.pad(x)))
#             LH, HL, HH = skips['pool3']
#             original = skips['conv3_4'] if 'conv3_4' in skips.keys() else None
#             out = self.recon_block3(out, LH, HL, HH, original)
#             _conv3_4 = self.conv3_4 if self.option_unpool == 'sum' else self.conv3_4_2
#             out = self.relu(_conv3_4(self.pad(out)))
#             out = self.relu(self.conv3_3(self.pad(out)))
#             return self.relu(self.conv3_2(self.pad(out)))
#         elif level == 3:
#             out = self.relu(self.conv3_1(self.pad(x)))
#             LH, HL, HH = skips['pool2']
#             original = skips['conv2_2'] if 'conv2_2' in skips.keys() else None
#             out = self.recon_block2(out, LH, HL, HH, original)
#             _conv2_2 = self.conv2_2 if self.option_unpool == 'sum' else self.conv2_2_2
#             return self.relu(_conv2_2(self.pad(out)))
#         elif level == 2:
#             out = self.relu(self.conv2_1(self.pad(x)))
#             LH, HL, HH = skips['pool1']
#             original = skips['conv1_2'] if 'conv1_2' in skips.keys() else None
#             out = self.recon_block1(out, LH, HL, HH, original)
#             _conv1_2 = self.conv1_2 if self.option_unpool == 'sum' else self.conv1_2_2
#             return self.relu(_conv1_2(self.pad(out)))
#         else:
#             return self.conv1_1(self.pad(x))
