# coding:utf8

import torch as t
import torchvision as tv
import torchnet as tnt

from torch.utils import data
from transformer_net import TransformerNet
import utils
from PackedVGG import Vgg16
from torch.nn import functional as F
import tqdm
import os
import ipdb
# from WCT2_train import WCT2
# import model

from LapSobGaus_train import Lap_Sob_Gaus
import net

import Ovodus_Laplace_model
import utils_
from WCT2_train import train_transform
from tensorboardX import SummaryWriter

from pathlib import Path

from torchvision.utils import save_image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Config(object):
    # General Args
    use_gpu = True
    model_path = None  # pretrain model path (for resume training or test)

    # Train Args
    image_size = 448  # image crop_size for training
    batch_size = 2
    data_root = r'F:\DataSets\train2017'  # 'data/'  dataset root：$data_root/coco/a.jpg  D:\CoCo_Dataset\train2017
    num_workers = 4  # dataloader num of workers

    lr = 1e-4
    epoches = 20  # total epoch to train
    content_weight = 1e10  # weight of content_loss
    style_weight = 1e2  # weight of style_loss

    style_path = 'style_input'  # style image path
    env = 'onlyencodercontent_58_Laps_test_nores_noDynamic_10_2'  # visdom env
    plot_every = 1  # visualize in visdom for every 10 batch

    debug_file = '/tmp/debugnn'  # touch $debug_fie to interrupt and enter ipdb

    # Test Args
    content_path = 'input.png'  # input file to do style transfer [for test]
    result_path = 'output.png'  # style transfer result [for test]

    option_unpool = 'sum'
    cpu = False
    transfer_at_encoder = True
    transfer_at_decoder = True
    transfer_at_skip = True
    verbose = True
    save_dir = './onlyencodercontent/nores_noDynamic/58_LapSobGaus_experiments_10_2'
    log_dir = './onlyencodercontent/nores_noDynamic/58_LapSobGaus_logs_10_2'

    lr_decay = 5e-5


def adjust_learning_rate(lr ,optimizer, iteration_count, lr_decay):
    """Imitating the original implementation"""
    lr = lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(**kwargs):
    opt = Config()
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = 'cpu' if opt.cpu or not t.cuda.is_available() else 'cuda:0'
    device = t.device(device)
    # device=t.device('cuda') if opt.use_gpu else t.device('cpu')
    vis = utils_.Visualizer(opt.env)

    save_dir = Path(opt.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path(opt.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    # Data loading
    transfroms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        #tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        #tv.transforms.Lambda(lambda x: x*255)
    ])
    dataset = tv.datasets.ImageFolder(opt.data_root, transfroms)
    dataloader = data.DataLoader(dataset, opt.batch_size)

    # style transformer network
    # transformer = TransformerNet()
    print('come!')
    # visualizer = Visualizer(config)  # create a visualizer that display/save images and plots
    # device = 'cpu' if opt.cpu or not t.cuda.is_available() else 'cuda:0'
    # device = t.device(device)

    transfer_at = set()
    if opt.transfer_at_encoder:
        transfer_at.add('encoder')
    if opt.transfer_at_decoder:
        transfer_at.add('decoder')
    if opt.transfer_at_skip:
        transfer_at.add('skip')
    # save_dir = Path(config.save_dir)
    # save_dir.mkdir(exist_ok=True, parents=True)
    # log_dir = Path(config.log_dir)
    # log_dir.mkdir(exist_ok=True, parents=True)
    # writer = SummaryWriter(log_dir=str(log_dir))
    # vgg = net.vgg
    wct2 = Lap_Sob_Gaus(transfer_at=transfer_at, option_unpool=opt.option_unpool, device=device,
                        verbose=False)

    encoder = Ovodus_Laplace_model.Lap_Sob_GausEncoder(opt.option_unpool).to(device)
    decoder = Ovodus_Laplace_model.Lap_Sob_GausDecoder(opt.option_unpool).to(device)
    # vgg.load_state_dict(torch.load(config.vgg))
    # vgg = nn.Sequential(*list(vgg.children())[:31])
    laps = Lap_Sob_Gaus(transfer_at=transfer_at, option_unpool='sum', device=device)
    network = net.Net(encoder, decoder)
    network.train()
    network.to(device)
    transformer = network
    if opt.model_path:
        transformer.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    transformer.to(device)

    # Vgg16 for Perceptual Loss
    # vgg = Vgg16().eval()
    # vgg.to(device)
    # for param in vgg.parameters():
    #     param.requires_grad = False

    # Optimizer
    # optimizer = t.optim.Adam(transformer.parameters(), opt.lr)
    enoptimizer = t.optim.Adam(network.encoder.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    deoptimizer = t.optim.Adam(network.decoder.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    # # Get style image
    # style_dataloader = utils_.get_style_data(opt.style_path, opt.batch_size)
    # #style_list = list(enumerate(style_dataloader))
    # for ii, (style, _) in tqdm.tqdm(enumerate(style_dataloader)):
    #     #a = style
    #     style = style.expand(opt.batch_size, 3, 256, 256)
    #     vis.img('style', (style.data[0] * 0.225 + 0.45).clamp(min=0, max=1))
    #     #style_list.append(style)
    #
    # style = style.to(device)
    # #
    # # #
    # # # # gram matrix for style image
    # with t.no_grad():
    #     features_style = vgg(style)
    #     gram_style = [utils_.gram_matrix(y) for y in features_style]

    # Loss meter
    style_meter = tnt.meter.AverageValueMeter()
    content_meter = tnt.meter.AverageValueMeter()

    for epoch in range(opt.epoches):
        # for jj, (style, _) in tqdm.tqdm(enumerate(style_dataloader)):
        #     a = style
        #     vis.img('style', (style.data[0] * 0.225 + 0.45).clamp(min=0, max=1))
        #     style = style.to(device)

        #


        content_meter.reset()
        style_meter.reset()
        for ii, (x, _) in tqdm.tqdm(enumerate(dataloader)):
            if epoch == 0:
                adjust_learning_rate(opt.lr, enoptimizer, iteration_count=ii,  lr_decay=opt.lr_decay)
                adjust_learning_rate(opt.lr, deoptimizer, iteration_count=ii, lr_decay=opt.lr_decay)
                print(opt.lr)
            # style = style_list[ii][1][0]
            # # style = style_list[ii]
            # style = style.to(device)
            # # # gram matrix for style image
            # with t.no_grad():
            #     features_style = vgg(style)
            #     gram_style = [utils_.gram_matrix(y) for y in features_style]
            style_dataloader = utils_.get_style_data(opt.style_path, opt.batch_size)
            # style_list = list(enumerate(style_dataloader))
            for jj, (style, _) in tqdm.tqdm(enumerate(style_dataloader)):
                # a = style
                style = style.expand(opt.batch_size, 3, 256, 256)
                #vis.img('style', (style.data[0] * 0.225 + 0.45).clamp(min=0, max=1))
                vis.img('style', (style.data[0]).clamp(min=0, max=1))
                # style_list.append(style)

            style = style.to(device)
            #
            # #
            # # # gram matrix for style image
            # with t.no_grad():
            #     features_style = vgg(style)
            #     gram_style = [utils_.gram_matrix(y) for y in features_style]
            # Train
            enoptimizer.zero_grad()
            deoptimizer.zero_grad()
            x = x.to(device)
            #y = network(x, style, Laps=laps)
            # if (ii + 1) % 10 == 0:
            # print(y)
            # y = y.clamp_(0, 1) * 255
            #y = utils_.normalize_batch(y)
            #x = utils_.normalize_batch(x)

            # features_y = vgg(y)
            # features_x = vgg(x)

            # # content loss
            # content_loss = opt.content_weight * F.mse_loss(features_y.relu2_2, features_x.relu2_2)
            #
            # # style loss
            # style_loss = 0
            #
            # for ft_y, gm_s in zip(features_y, gram_style):
            #     gram_y = utils_.gram_matrix(ft_y)
            #     style_loss += F.mse_loss(gram_y, gm_s.expand_as(gram_y))
            y, content_feats, content_loss, style_loss = network(x, style, Laps=laps)
            content_loss *= opt.content_weight
            style_loss *= opt.style_weight
            total_loss = content_loss + style_loss
            total_loss.backward()
            enoptimizer.step()
            deoptimizer.step()

            # Loss smooth for visualization
            content_meter.add(content_loss.item())
            style_meter.add(style_loss.item())

            if ii % 50 == 1:
                print('\n')
                print('iters:', ii, 'total_loss:', total_loss, 'loss_c:', content_loss, 'loss_s: ', style_loss)
            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # visualization
                vis.plot('content_loss', content_meter.value()[0])
                vis.plot('style_loss', style_meter.value()[0])
                # denorm input/output, since we have applied (utils.normalize_batch)
                vis.img('output1', (y.data.cpu()[0]).clamp(min=0, max=1))
                vis.img('input1', (x.data.cpu()[0]).clamp(min=0, max=1))
                vis.img('decoder_1', (content_feats['decoder'][0][0].data.cpu()[0]).clamp(min=0, max=1))
                vis.img('decoder_2', (content_feats['decoder'][1][0].data.cpu()[0]).clamp(min=0, max=1))
                vis.img('decoder_3', (content_feats['decoder'][2][0].data.cpu()[0]).clamp(min=0, max=1))
                vis.img('decoder_4', (content_feats['decoder'][3][0].data.cpu()[0]).clamp(min=0, max=1))
                #save_image(content_feat.clamp_(0, 1), fname_output + "decoder{:d}".format(level), padding=0)

            if (ii) % 1000 == 0:
                if not os.path.exists(save_dir /'epoch_{:d}'.format(epoch)):
                    os.makedirs(save_dir /'epoch_{:d}'.format(epoch))
                de_state_dict = network.decoder.state_dict()
                en_state_dict = network.encoder.state_dict()
                for key in de_state_dict.keys():
                    de_state_dict[key] = de_state_dict[key].to(t.device('cpu'))
                t.save(de_state_dict, save_dir /'epoch_{:d}'.format(epoch)/
                       'decoder_iter_{:d}.pth.tar'.format(ii + 1))
                for key in en_state_dict.keys():
                    en_state_dict[key] = en_state_dict[key].to(t.device('cpu'))
                t.save(en_state_dict, save_dir /'epoch_{:d}'.format(epoch)/
                       'encoder_iter_{:d}.pth.tar'.format(ii + 1))

        de_state_dict = network.decoder.state_dict()
        en_state_dict = network.encoder.state_dict()
        for key in de_state_dict.keys():
            de_state_dict[key] = de_state_dict[key].to(t.device('cpu'))
        t.save(de_state_dict, save_dir /
               'epoch_decoder_iter_{:d}.pth.tar'.format(epoch + 1))
        for key in en_state_dict.keys():
            en_state_dict[key] = en_state_dict[key].to(t.device('cpu'))
        t.save(en_state_dict, save_dir /
               'epoch_encoder_iter_{:d}.pth.tar'.format(epoch + 1))
        # save checkpoints
        vis.save([opt.env])
        t.save(network.state_dict(), 'checkpoints/epoch_%s_style.pth' % epoch)
    writer.close()


@t.no_grad()
def stylize(**kwargs):
    """
    perform style transfer
    """
    opt = Config()

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')

    # input image preprocess
    content_image = tv.datasets.folder.default_loader(opt.content_path)
    content_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device).detach()

    # model setup
    style_model = TransformerNet().eval()
    style_model.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    style_model.to(device)

    # style transfer and save output
    output = style_model(content_image)
    output_data = output.cpu().data[0]
    tv.utils.save_image(((output_data / 255)).clamp(min=0, max=1), opt.result_path)


if __name__ == '__main__':
    import fire

    fire.Fire()
    train()