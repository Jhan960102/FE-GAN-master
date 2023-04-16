"""
 > Training pipeline for FE-GAN model

"""
# py libs
import os
import sys
import yaml
import argparse
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
# local libs
from nets.commons import Weights_Normal, VGG19_PercepLoss
from nets.fegan import Generator, Discriminator
from utils.data_utils import GetTrainingPairs, GetValImage

if __name__ == '__main__':

    # get configs and training options
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="./configs/train_euvp.yaml")
    parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
    parser.add_argument("--num_epochs", type=int, default=201, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of 1st order momentum")
    parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of 2nd order momentum")
    args = parser.parse_args()

    # training params
    epoch = args.epoch
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr_rate, lr_b1, lr_b2 = args.lr, args.b1, args.b2

    # load the data config file
    with open(args.cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # get info from config file
    dataset_name = cfg["dataset_name"]
    dataset_path = cfg["dataset_path"]
    channels = cfg["chans"]
    img_width = cfg["im_width"]
    img_height = cfg["im_height"]
    val_interval = cfg["val_interval"]
    ckpt_interval = cfg["ckpt_interval"]

    # create dir for model and validation data
    samples_dir = os.path.join("./samples/FE-GAN/", dataset_name)
    checkpoint_dir = os.path.join("./checkpoints/FE-GAN/", dataset_name)         # model save_dir

    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)


    """ FE-GAN specifics: loss functions and patch-size
    -----------------------------------------------------"""
    Adv_cGAN = torch.nn.MSELoss().cuda()
    L1_G = torch.nn.L1Loss().cuda()                          # similarity loss (l1_loss)
    L_vgg = VGG19_PercepLoss().cuda()                        # content loss (perceptual_loss)
    lambda_1, lambda_con = 7, 3                         # 7:3-----paired training
    patch = (1, img_height//16, img_width//16)          # 16x16 for 256x256

    # Initialize generator and discriminator
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()

    is_cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor

    # Initialize weights or load pretrained models
    if args.epoch == 0:
        generator.apply(Weights_Normal)
        discriminator.apply(Weights_Normal)
    else:
        generator.load_state_dict(torch.load("./checkpoints/FE-GAN/%s/generator_%d.pth" % (dataset_name, args.epoch)))
        discriminator.load_state_dict(torch.load("./checkpoints/FE-GAN/%s/discriminator_%d.pth" % (dataset_name, epoch)))
        print("Loaded model from epoch %d" %(epoch))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))


    ## Data pipeline
    transforms_ = [
        transforms.Resize((img_height, img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataloader = DataLoader(
        GetTrainingPairs(dataset_path, dataset_name, transforms_=transforms_),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    val_dataloader = DataLoader(
        GetValImage(dataset_path, dataset_name, transforms_=transforms_, sub_dir='validation'),
        batch_size=4,
        shuffle=True,
        num_workers=1,
    )

    # create a .csv file to record loss
    df = pd.DataFrame(columns=['time', 'epoch', 'loss_D', 'loss_G', 'loss_GAN'])
    df.to_csv("./loss/FE-GAN_loss.csv", mode='a+', index=False)

    ## Training pipeline
    for epoch in range(epoch, num_epochs):
        for i, batch in enumerate(dataloader):
            # Model inputs
            imgs_distorted = Variable(batch["A"].type(Tensor))
            imgs_good_gt = Variable(batch["B"].type(Tensor))
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_distorted.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_distorted.size(0), *patch))), requires_grad=False)

            # Train Discriminator
            optimizer_D.zero_grad()
            imgs_fake = generator(imgs_distorted)
            pred_real = discriminator(imgs_good_gt, imgs_distorted)
            loss_real = Adv_cGAN(pred_real, valid)
            pred_fake = discriminator(imgs_fake, imgs_distorted)
            loss_fake = Adv_cGAN(pred_fake, fake)
            # Total loss: real + fake (standard PatchGAN)
            loss_D = 0.5 * (loss_real + loss_fake) * 10.0       # 10x scaled for stability
            loss_D.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            imgs_fake = generator(imgs_distorted)
            pred_fake = discriminator(imgs_fake, imgs_distorted)
            loss_GAN = Adv_cGAN(pred_fake, valid)                   # GAN loss
            loss_1 = L1_G(imgs_fake, imgs_good_gt)                  # similarity loss
            loss_con = L_vgg(imgs_fake, imgs_good_gt)               # content loss

            # Total loss        forward + backward + update
            loss_G = loss_GAN + lambda_1 * loss_1 + lambda_con * loss_con
            loss_G.backward()
            optimizer_G.step()

            # Print log
            if not i % 50:
                sys.stdout.write("\r[Epoch %d/%d: batch %d/%d] [DLoss: %.3f, GLoss: %.3f, AdvLoss: %.3f]"
                                  %(
                                    epoch, num_epochs, i, len(dataloader),
                                    loss_D.item(), loss_G.item(), loss_GAN.item(),
                                   )
                )

            # If at sample interval save image
            batches_done = epoch * len(dataloader) + i
            if batches_done % val_interval == 0:
                imgs = next(iter(val_dataloader))
                imgs_val = Variable(imgs["val"].type(Tensor))
                imgs_gen = generator(imgs_val)
                img_sample = torch.cat((imgs_val.data, imgs_gen.data), -2)
                save_image(img_sample, "./FE-GAN/%s/%s.png" % (dataset_name, batches_done), nrow=5, normalize=True)

            if (epoch % ckpt_interval == 0):
                # save the data in a one-dimensional list
                time = "%s" % datetime.now()
                step = "Step[%d]" % epoch
                loss_list = [time, step, loss_D.item(), loss_G.item(), loss_GAN.item()]

                data = pd.DataFrame([loss_list])
                data.to_csv("./loss/FE-GAN_loss.csv", mode='a+', header=False,
                            index=False)

        # save model checkpoints
        if (epoch % ckpt_interval == 0):
            torch.save(generator.state_dict(), "./checkpoints/FE-GAN/%s/generator_%d.pth" % (dataset_name, epoch))
            torch.save(discriminator.state_dict(), "./checkpoints/FE-GAN/%s/discriminator_%d.pth" % (dataset_name, epoch))
