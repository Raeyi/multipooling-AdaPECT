import torch.nn as nn
from utils.core import feature_wct
import torch

import utils_
from function import adaptive_instance_normalization as adain
# from WCT2_train import WCT2
#from LapSobGaus_train import Lap_Sob_Gaus
from function import calc_mean_std

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class Net(nn.Module):
    def __init__(self, encoder, decoder, device='cuda:0', verbose=False, vgg=None, transfer_at=['encoder', 'skip', 'decoder']):
        super(Net, self).__init__()
        # self.transfer_at = set(transfer_at)
        # assert not (self.transfer_at - set(['encoder', 'decoder', 'skip'])), 'invalid transfer_at: {}'.format(
        #     transfer_at)
        # assert self.transfer_at, 'empty transfer_at'
        # enc_layers = list(vgg.children())
        # self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        # self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        # self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        # self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.transfer_at = transfer_at
        self.decoder = decoder
        self.verbose = verbose
        self.encoder = encoder
        self.mse_loss = nn.MSELoss()
        self.device = torch.device(device)

        # # fix the encoder
        # for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
        #     for param in getattr(self, name).parameters():
        #         param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    # def encode_with_intermediate(self, x):
    #     results = [x]
    #     for i in range(4):
    #         func = getattr(self, 'enc_{:d}'.format(i + 1))
    #         results.append(func(results[-1]))
    #     return results[1:]
    #
    # # extract relu4_1 from input image
    # def encode(self, x):
    #     for i in range(4):
    #         x = getattr(self, 'enc_{:d}'.format(i + 1))(x)
    #     return x

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def get_all_feature(self, x):
        skips = {}
        feats = {'encoder': {}, 'decoder': {}}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            if 'encoder' in self.transfer_at:
                feats['encoder'][level] = x

        if 'encoder' not in self.transfer_at:
            feats['decoder'][4] = x
        for level in [4, 3, 2, 1]:
            x = self.decode(x, skips, level)
            if 'decoder' in self.transfer_at:
                feats['decoder'][level - 1] = x
        return x, feats, skips

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)
    def forward(self, content, style, wct2=None, Laps = None, alpha=1.0):
        assert 0 <= alpha <= 1
        # content_feat = wct2.transfer(content, style, content_segment=None, style_segment=None)
        #label_set, label_indicator = compute_label_info(content_segment, style_segment)
        # content_feat, content_skips = content, {}




        # style, style_skips = self.get_all_feature(style)
        # wct2_enc_level = [1, 2, 3, 4]
        # wct2_dec_level = [1, 2, 3, 4]
        # wct2_skip_level = ['pool1', 'pool2', 'pool3']
        # content_feat = content
        # content_skips = {}
        # #content_feats = {'encoder': {}, 'decoder': {}}
        # style_feats = {'encoder': {}, 'decoder': {}}
        content_skips, content_feature = {}, content
        for level in [1, 2, 3, 4]:
            #style = self.encode(style, style_skips, level)
            content_feature = self.encode(content_feature, content_skips, level)

        #     if 'encoder' in self.transfer_at:
        #         #content_feats['encoder'][level] = content
        #         style_feats['encoder'][level] = style
        content_feat, content_feats, __ = self.get_all_feature(content)
        # #
        # # if 'skip' in self.transfer_at:
        # #     for skip_level in wct2_skip_level:
        # #         for component in [0, 1, 2]:  # component: [LH, HL, HH]
        # #             content_skips[skip_level][component] = feature_wct(content_skips[skip_level][component], style_skips[skip_level][component],
        # #                                                                content_segment=None, style_segment=None,
        # #                                                                label_set=None, label_indicator=None,
        # #                                                                alpha=alpha, device=self.device)
        # for level in [4, 3, 2, 1]:
        #     #style = self.decode(style, style_skips, level)
        #     content_feat = self.decode(content_feat, content_skips, level)
        #content_feat = Laps.transfer(content, style, content_segment=None, style_segment=None, alpha=1)

        transfer_feat, transfer_skips = content_feat, {}
        for level in [1, 2, 3, 4]:
            transfer_feat = self.encode(transfer_feat, transfer_skips, level)
        content_loss = self.mse_loss(content_feat, content)
        style_loss = torch.zeros(1).to(self.device)
        style_loss += self.mse_loss(transfer_feat, content_feature.detach())


        return content_feat, content_feats, content_loss, style_loss
        #return content_feat['decoder'][0]
            #if 'decode' in self.transfer_at:
                #content_feats['decoder'][level - 1] = content


        # style_feats, style_skips = self.get_all_feature(style)
        #
        # wct2_enc_level = [1, 2, 3, 4]
        # wct2_dec_level = [1, 2, 3, 4]
        # wct2_skip_level = ['pool1', 'pool2', 'pool3']
        #
        # for level in [1, 2, 3, 4]:
        #     content_feat = self.encode(content_feat, content_skips, level)
        #     if 'encoder' in self.transfer_at and level in wct2_enc_level:
        #         content_feat = feature_wct(content_feat, style_feats['encoder'][level],
        #                                    content_segment=None, style_segment=None,
        #                                    label_set=None, label_indicator=None,
        #                                    alpha=alpha, device=self.device)
        # if 'skip' in self.transfer_at:
        #     for skip_level in wct2_skip_level:
        #         for component in [0, 1, 2]:  # component: [LH, HL, HH]
        #             content_skips[skip_level][component] = feature_wct(content_skips[skip_level][component],
        #                                                                style_skips[skip_level][component],
        #                                                                content_segment=None, style_segment=None,
        #                                                                label_set=None, label_indicator=None,
        #                                                                alpha=alpha, device=self.device)
        #
        # for level in [4, 3, 2, 1]:
        #     if 'decoder' in self.transfer_at and level in style_feats['decoder'] and level in wct2_dec_level:
        #         content_feat = feature_wct(content_feat, style_feats['decoder'][level],
        #                                    content_segment=None, style_segment=None,
        #                                    label_set=None, label_indicator=None,
        #                                    alpha=alpha, device=self.device)
        #     content_feat = self.decode(content_feat, style_skips, level)
        #content_feat = self.decode(content_feat, content_skips, 4)

