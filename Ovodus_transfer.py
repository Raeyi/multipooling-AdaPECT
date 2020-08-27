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

import os
import tqdm
import argparse

import utils_

import torch
import torchvision as tv
from torchvision.utils import save_image

from model import WaveEncoder, WaveDecoder
from torch.utils import data
from torchvision import transforms
from PIL import Image

from utils.core import feature_wct
from utils.io import Timer, open_image, load_segment, compute_label_info

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG',
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class WCT2:
    def __init__(self, model_path='.\experiments', transfer_at=['encoder', 'skip', 'decoder'],
                 option_unpool='cat5', device='cuda:0', verbose=False):

        self.transfer_at = set(transfer_at)
        assert not (self.transfer_at - set(['encoder', 'decoder', 'skip'])), 'invalid transfer_at: {}'.format(
            transfer_at)
        assert self.transfer_at, 'empty transfer_at'

        self.device = torch.device(device)
        self.verbose = verbose
        self.encoder = WaveEncoder(option_unpool).to(self.device)
        self.decoder = WaveDecoder(option_unpool).to(self.device)
        self.encoder.load_state_dict(
            torch.load(os.path.join(model_path, 'wave_encoder_{}_l4.pth'.format(option_unpool)),
                       map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(
            torch.load(os.path.join(model_path, 'wave_decoder_{}_l4.pth'.format(option_unpool)),
                       map_location=lambda storage, loc: storage))

    def print_(self, msg):
        if self.verbose:
            print(msg)

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def get_all_feature(self, x):
        skips = {}
        feats = {'encoder': {}, 'decoder': {}}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            if 'encoder' in self.transfer_at:
                feats['encoder'][level] = x

        if 'encoder' not in self.transfer_at:
            feats['decoder'][4] = x
        for level in [4, 3, 2]:
            x = self.decode(x, skips, level)
            if 'decoder' in self.transfer_at:
                feats['decoder'][level - 1] = x
        return feats, skips

    def transfer(self, content, style, content_segment, style_segment, alpha=1):
        label_set, label_indicator = compute_label_info(content_segment, style_segment)
        content_feat, content_skips = content, {}
        style_feats, style_skips = self.get_all_feature(style)

        wct2_enc_level = [1, 2, 3, 4]
        wct2_dec_level = [1, 2, 3, 4]
        wct2_skip_level = ['pool1', 'pool2', 'pool3']

        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            if 'encoder' in self.transfer_at and level in wct2_enc_level:
                content_feat = feature_wct(content_feat, style_feats['encoder'][level],
                                           content_segment, style_segment,
                                           label_set, label_indicator,
                                           alpha=alpha, device=self.device)
                self.print_('transfer at encoder {}'.format(level))
        if 'skip' in self.transfer_at:
            for skip_level in wct2_skip_level:
                for component in [0, 1, 2]:  # component: [LH, HL, HH]
                    content_skips[skip_level][component] = feature_wct(content_skips[skip_level][component],
                                                                       style_skips[skip_level][component],
                                                                       content_segment, style_segment,
                                                                       label_set, label_indicator,
                                                                       alpha=alpha, device=self.device)
                self.print_('transfer at skip {}'.format(skip_level))

        for level in [4, 3, 2, 1]:
            if 'decoder' in self.transfer_at and level in style_feats['decoder'] and level in wct2_dec_level:
                content_feat = feature_wct(content_feat, style_feats['decoder'][level],
                                           content_segment, style_segment,
                                           label_set, label_indicator,
                                           alpha=alpha, device=self.device)
                self.print_('transfer at decoder {}'.format(level))
            content_feat = self.decode(content_feat, content_skips, level)
        return content_feat

    def transfer1(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        content_feat = content
        content_skips, style_skips = {}, {}
        style_feats = {'encoder': {}, 'decoder': {}}
        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            style = self.encode(style, style_skips, level)
            if 'encoder' in self.transfer_at:
                # content_feats['encoder'][level] = content
                style_feats['encoder'][level] = style
        for level in [4, 3, 2, 1]:
            content_feat = self.decode(content_feat, style_skips, level)

        img = utils_.normalize_batch(content_feat)
        img = img.data.cpu()[0] * 0.225 + 0.45
        return img


def get_all_transfer():
    ret = []
    for e in ['encoder', None]:
        for d in ['decoder', None]:
            for s in ['skip', None]:
                _ret = set([e, d, s]) & set(['encoder', 'decoder', 'skip'])
                if _ret:
                    ret.append(_ret)
    return ret


def run_bulk(config):
    device = 'cpu' if config.cpu or not torch.cuda.is_available() else 'cuda:0'
    device = torch.device(device)

    transfer_at = set()
    if config.transfer_at_encoder:
        transfer_at.add('encoder')
    if config.transfer_at_decoder:
        transfer_at.add('decoder')
    if config.transfer_at_skip:
        transfer_at.add('skip')

    # The filenames of the content and style pair should match
    fnames = set(os.listdir(config.content)) & set(os.listdir(config.style))

    if config.content_segment and config.style_segment:
        fnames &= set(os.listdir(config.content_segment))
        fnames &= set(os.listdir(config.style_segment))

    for fname in tqdm.tqdm(fnames):
        if not is_image_file(fname):
            print('invalid file (is not image), ', fname)
            continue
        _content = os.path.join(config.content, fname)
        _style = os.path.join(config.style, fname)
        _content_segment = os.path.join(config.content_segment, fname) if config.content_segment else None
        _style_segment = os.path.join(config.style_segment, fname) if config.style_segment else None
        _output = os.path.join(config.output, fname)

        # content_transfroms = tv.transforms.Compose([
        #     #tv.transforms.Resize(config.image_size),
        #     #tv.transforms.CenterCrop(config.image_size),
        #     tv.transforms.ToTensor(),
        #     # tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        #     tv.transforms.Lambda(lambda x: x * 255)
        # ])
        # content_dataset = tv.datasets.ImageFolder(_content, content_transfroms)
        # content_dataloader = data.DataLoader(content_dataset, 1)
        #
        # style_transform = tv.transforms.Compose([
        #     # tv.transforms.Resize(256),
        #     # tv.transforms.RandomCrop(256),
        #     tv.transforms.ToTensor(),
        #     tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        #     # tv.transforms.Lambda(lambda x: x*255)
        # ])
        #
        # style_image_dataset = tv.datasets.ImageFolder(_style, style_transform)
        #
        # style_dataloader = data.DataLoader(style_image_dataset, 1)

        #content = open_image(_content, config.image_size).to(device)
        content = Image.open(_content)
        content_transforms = []
        content = transforms.Resize(config.image_size)(content)
            # _transforms.append(transforms.Resize(image_size))
        w, h = content.size
        content_transforms.append(transforms.CenterCrop((h // 16 * 16, w // 16 * 16)))
        content_transforms.append(transforms.ToTensor())
        content_transforms.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        content_transforms.append(transforms.Lambda(lambda x: x * 255))
        content_transform = transforms.Compose(content_transforms)
        content = content_transform(content).unsqueeze(0).to(device)

        #style = open_image(_style, config.image_size).to(device)
        style = Image.open(_style)
        style_transforms = []
        style = transforms.Resize(config.image_size)(style)
        # _transforms.append(transforms.Resize(image_size))
        w, h = style.size
        style_transforms.append(transforms.CenterCrop((h // 16 * 16, w // 16 * 16)))
        style_transforms.append(transforms.ToTensor())
        style_transforms.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        #style_transforms.append(transforms.Lambda(lambda x: x * 255))
        style_transform = transforms.Compose(style_transforms)
        style = style_transform(style).unsqueeze(0).to(device)


        content_segment = load_segment(_content_segment, config.image_size)
        style_segment = load_segment(_style_segment, config.image_size)
        _, ext = os.path.splitext(fname)

        if not config.transfer_all:
            with Timer('Elapsed time in whole WCT: {}', config.verbose):
                postfix = '_'.join(sorted(list(transfer_at)))
                fname_output = _output.replace(ext, '_{}_{}.{}'.format(config.option_unpool, postfix, ext))
                print('------ transfer:', _output)
                wct2 = WCT2(transfer_at=transfer_at, option_unpool=config.option_unpool, device=device,
                            verbose=config.verbose)
                with torch.no_grad():
                    img = wct2.transfer(content, style, content_segment, style_segment, alpha=config.alpha)
                save_image(img.clamp_(0, 1), fname_output, padding=0)
        else:
            for _transfer_at in get_all_transfer():
                with Timer('Elapsed time in whole WCT: {}', config.verbose):
                    postfix = '_'.join(sorted(list(_transfer_at)))
                    fname_output = _output.replace(ext, '_{}_{}.{}'.format(config.option_unpool, postfix, ext))
                    print('------ transfer:', fname)
                    wct2 = WCT2(transfer_at=_transfer_at, option_unpool=config.option_unpool, device=device,
                                verbose=config.verbose)
                    with torch.no_grad():
                        img = wct2.transfer(content, style, content_segment=content_segment,
                                            style_segment=style_segment, alpha=config.alpha)
                        img = utils_.normalize_batch(img)
                        #img = img.data.cpu()[0] * 0.225 + 0.45
                    save_image(img.clamp_(0, 1), fname_output, padding=0)
                    # torch.transforms.Resize(256,256),
                    # with torch.no_grad():
                    #     img = wct2.transfer1(content, style)
                    # save_image(img.clamp_(0, 1), fname_output, padding=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='./examples/content')
    parser.add_argument('--content_segment', type=str, default=None)
    parser.add_argument('--style', type=str, default='./examples/style')
    parser.add_argument('--style_segment', type=str, default=None)
    parser.add_argument('--output', type=str, default='./outputs')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--option_unpool', type=str, default='cat5', choices=['sum', 'cat5'])
    parser.add_argument('-e', '--transfer_at_encoder', action='store_true')
    parser.add_argument('-d', '--transfer_at_decoder', action='store_true')
    parser.add_argument('-s', '--transfer_at_skip', action='store_true')
    parser.add_argument('-a', '--transfer_all', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    config = parser.parse_args()

    print(config)

    if not os.path.exists(os.path.join(config.output)):
        os.makedirs(os.path.join(config.output))

    '''
    CUDA_VISIBLE_DEVICES=6 python transfer.py --content ./examples/content --style ./examples/style --content_segment ./examples/content_segment --style_segment ./examples/style_segment/ --output ./outputs/ --verbose --image_size 512 -a
    '''
    run_bulk(config)
