"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python WCT2_train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python WCT2_train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import os
from tqdm import tqdm
import argparse
from pathlib import Path
import utils_

from visdom import Visdom
import numpy as np

import torchnet as tnt
#from visualizer import Visualizer

from PIL import Image, ImageFile

import torch
from torchvision.utils import save_image
import torch.nn as nn
from torchvision import transforms
from torch.utils import data

#from model import WaveEncoder, WaveDecoder
from Ovodus_Laplace_model import Lap_Sob_GausEncoder, Lap_Sob_GausDecoder

from tensorboardX import SummaryWriter

#from utils.core import feature_wct
from AdWCT import feature_wct
from utils.io import Timer, open_image, load_segment, compute_label_info

import torchvision as tv

from function import adaptive_instance_normalization as adain
from function import calc_mean_std

import net

from sampler import InfiniteSamplerWrapper

import utils
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG',
]



class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class Lap_Sob_Gaus:
    def __init__(self, transfer_at=['encoder', 'skip', 'decoder'], option_unpool='cat5', device='cuda:0', verbose=False, vgg=None):

        self.transfer_at = set(transfer_at)
        assert not (self.transfer_at - set(['encoder', 'decoder', 'skip'])), 'invalid transfer_at: {}'.format(
            transfer_at)
        assert self.transfer_at, 'empty transfer_at'
        # enc_layers = list(vgg.children())
        # self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        # self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        # self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        # self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.device = torch.device(device)
        self.verbose = verbose
        self.encoder = Lap_Sob_GausEncoder(option_unpool).to(self.device)
        self.decoder = Lap_Sob_GausDecoder(option_unpool).to(self.device)
        self.mse_loss = nn.MSELoss()
        for name in ['encoder', 'decoder']:
            for param in getattr(self, name).parameters():
                param.requires_grad = True
        # self.encoder.load_state_dict(
        #     torch.load(os.path.join(model_path, 'wave_encoder_{}_l4.pth'.format(option_unpool)),
        #                map_location=lambda storage, loc: storage))
        # self.decoder.load_state_dict(
        #     torch.load(os.path.join(model_path, 'wave_decoder_{}_l4.pth'.format(option_unpool)),
        #                map_location=lambda storage, loc: storage))

    def print_(self, msg):
        if self.verbose:
            print(msg)

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, x):
        results = [x]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def vgg_encode(self, x):
        for i in range(4):
            x = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return x

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
                                           label_set=None, label_indicator=None,
                                           alpha=alpha, device=self.device)
                self.print_('transfer at encoder {}'.format(level))
        if 'skip' in self.transfer_at:
            for skip_level in wct2_skip_level:
                for component in [0, 1, 2]:  # component: [LH, HL, HH]
                    content_skips[skip_level][component] = feature_wct(content_skips[skip_level][component],
                                                                       style_skips[skip_level][component],
                                                                       content_segment, style_segment,
                                                                       label_set=None, label_indicator=None,
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






def get_all_transfer():
    ret = []
    for e in ['encoder', None]:
        for d in ['decoder', None]:
            for s in ['skip', None]:
                _ret = set([e, d, s]) & set(['encoder', 'decoder', 'skip'])
                if _ret:
                    ret.append(_ret)
    return ret

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = config.lr / (1.0 + config.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def run_train(config):
    print('come!')
    #visualizer = Visualizer(config)  # create a visualizer that display/save images and plots
    device = 'cpu' if config.cpu or not torch.cuda.is_available() else 'cuda:0'
    device = torch.device(device)

    transfer_at = set()
    if config.transfer_at_encoder:
        transfer_at.add('encoder')
    if config.transfer_at_decoder:
        transfer_at.add('decoder')
    if config.transfer_at_skip:
        transfer_at.add('skip')
    save_dir = Path(config.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path(config.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    vgg = net.vgg
    wct2 = Lap_Sob_Gaus(transfer_at=transfer_at, option_unpool=config.option_unpool, device=device,
                                          verbose=config.verbose, vgg=vgg)

    encoder = Lap_Sob_GausEncoder(config.option_unpool).to(device)
    decoder = Lap_Sob_GausDecoder(config.option_unpool).to(device)
    vgg.load_state_dict(torch.load(config.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    network = net.Net(encoder, decoder, vgg=vgg)
    network.train()
    network.to(device)

    content_tf = train_transform()
    style_tf = train_transform()

    # # Data loading
    # transfroms = tv.transforms.Compose([
    #      tv.transforms.Resize(config.image_size),
    #      tv.transforms.CenterCrop(config.image_size),
    #      tv.transforms.ToTensor(),
    #      tv.transforms.Lambda(lambda x: x * 255)
    # ])
    # dataset = tv.datasets.ImageFolder(config.data_root, transfroms)
    # dataloader = data.DataLoader(dataset, config.batch_size)



    content_dataset = FlatFolderDataset(config.content_dir, content_tf)
    style_dataset = FlatFolderDataset(config.style_dir, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=config.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=config.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=config.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=config.n_threads))

    # Optimizer
    enoptimizer = torch.optim.Adam(network.encoder.parameters(), lr=config.lr)
    deoptimizer = torch.optim.Adam(network.decoder.parameters(), lr=config.lr)
    # # Loss meter
    # style_meter = tnt.meter.AverageValueMeter()
    # content_meter = tnt.meter.AverageValueMeter()
    vis = Visdom(env="loss")
    # style = utils.get_style_data(config.style_path)
    # vis.img('style', (style.data[0] * 0.225 + 0.45).clamp(min=0, max=1))
    # style = style.to(device)

    contet_loss, style_loss, iters = 0, 0, 0

    win_c = vis.line(np.array([contet_loss]), np.array([iters]), win='content_loss')
    win_s = vis.line(np.array([style_loss]), np.array([iters]), win='style_loss')
    # for epoch in range(config.epoches):
    #     content_meter.reset()
    #     style_meter.reset()
    #     for ii, (x, _) in tqdm.tqdm(enumerate(dataloader)):
            # Train
    for i in tqdm(range(config.max_iter)):
        enoptimizer.zero_grad()
        deoptimizer.zero_grad()
        # x = x.to(device)
        # y = network(x, style)


        adjust_learning_rate(enoptimizer, iteration_count=i)
        adjust_learning_rate(deoptimizer, iteration_count=i)
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device)
        content_images.requires_grad_()
        style_images.requires_grad_()
        loss_c, loss_s = network(content_images, style_images, wct2)
        loss_c = config.content_weight * loss_c
        loss_s = config.style_weight * loss_s
        loss = loss_c + loss_s

        # optimizer.zero_grad()
        loss.backward()
        enoptimizer.step()
        deoptimizer.step()

        if i % 50 == 1:
            print('\n')
            print('iters:', i, 'loss:', loss, 'loss_c:', loss_c, 'loss_s: ', loss_s)
        if i % 20 == 0:
            iters = np.array([i])
            content_loss = np.array([loss_c.item()])
            style_loss = np.array([loss_s.item()])
            vis.line(content_loss, iters, win_c, update='append')
            vis.line(style_loss, iters, win_s, update='append')

        writer.add_scalar('loss_content', loss_c.item(), i + 1)
        writer.add_scalar('loss_style', loss_s.item(), i + 1)

        if (i + 1) % config.save_model_interval == 0 or (i + 1) == config.max_iter:
            state_dict = network.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                       'decoder_iter_{:d}.pth.tar'.format(i + 1))
        if (i + 1) % config.save_model_interval == 0 or (i + 1) == config.max_iter:
            state_dict = network.encoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                       'encoder_iter_{:d}.pth.tar'.format(i + 1))
    writer.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', type=str, default=r'D:\wct2\WCT2-master\examples\content')
    parser.add_argument('--content_segment', type=str, default=None)
    parser.add_argument('--style_dir', type=str, default=r'D:\wct2\WCT2-master\examples\style1')
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
    parser.add_argument('--vgg', type=str, default='Vgg_models/vgg_normalised.pth')
    # training options
    parser.add_argument('--save_dir', default='./experiments',
                        help='Directory to save the model')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--style_weight', type=float, default=1.0)
    parser.add_argument('--content_weight', type=float, default=1000.0)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=1000)
    # visdom and HTML visualization parameters
    parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
    parser.add_argument('--display_ncols', type=int, default=4,
                        help='if positive, display all images in a single visdom web panel with certain number of images per row.')
    parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
    parser.add_argument('--display_server', type=str, default="http://localhost",
                        help='visdom server of the web display')
    parser.add_argument('--display_env', type=str, default='main',
                        help='visdom display environment name (default is "main")')
    parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
    parser.add_argument('--update_html_freq', type=int, default=1000,
                        help='frequency of saving training results to html')
    parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
    parser.add_argument('--no_html', action='store_true',
                        help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
    config = parser.parse_args()

    # print(config)

    if not os.path.exists(os.path.join(config.output)):
        os.makedirs(os.path.join(config.output))

    '''
    CUDA_VISIBLE_DEVICES=6 python transfer.py --content ./examples/content --style ./examples/style --content_segment ./examples/content_segment --style_segment ./examples/style_segment/ --output ./outputs/ --verbose --image_size 512 -a
    '''
    print('begin:')
    run_train(config)
