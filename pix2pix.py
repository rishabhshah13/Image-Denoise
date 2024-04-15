import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import center_crop
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.functional.image import peak_signal_noise_ratio



class DownSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True):
        """
        Paper details:
        - C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        """
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x
    

class UpSampleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=4,
        strides=2,
        padding=1,
        activation=True,
        batchnorm=True,
        dropout=False
    ):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.ReLU(True)

        if dropout:
            self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.deconv(x)
        if self.batchnorm:
            x = self.bn(x)

        if self.dropout:
            x = self.drop(x)
        return x
    


class Generator(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Paper details:
        - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        - Decoder: CD512-CD1024-CD1024-C1024-C1024-C512 -C256-C128
        """
        super().__init__()

        # encoder/donwsample convs
        self.encoders = [
            DownSampleConv(in_channels, 64, batchnorm=False),  # bs x 64 x 128 x 128
            DownSampleConv(64, 128),  # bs x 128 x 64 x 64
            DownSampleConv(128, 256),  # bs x 256 x 32 x 32
            DownSampleConv(256, 512),  # bs x 512 x 16 x 16
            DownSampleConv(512, 512),  # bs x 512 x 8 x 8
            DownSampleConv(512, 512),  # bs x 512 x 4 x 4
            DownSampleConv(512, 512),  # bs x 512 x 2 x 2
            DownSampleConv(512, 512, batchnorm=False),  # bs x 512 x 1 x 1
        ]

        # decoder/upsample convs
        self.decoders = [
            UpSampleConv(512, 512, dropout=True),  # bs x 512 x 2 x 2
            UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 4 x 4
            UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 8 x 8
            UpSampleConv(1024, 512),  # bs x 512 x 16 x 16
            UpSampleConv(1024, 256),  # bs x 256 x 32 x 32
            UpSampleConv(512, 128),  # bs x 128 x 64 x 64
            UpSampleConv(256, 64),  # bs x 64 x 128 x 128
        ]
        self.decoder_channels = [512, 512, 512, 512, 256, 128, 64]
        self.final_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        skips_cons = []
        for encoder in self.encoders:
            x = encoder(x)

            skips_cons.append(x)

        skips_cons = list(reversed(skips_cons[:-1]))
        decoders = self.decoders[:-1]

        for decoder, skip in zip(decoders, skips_cons):
            x = decoder(x)
            # print(x.shape, skip.shape)
            x = torch.cat((x, skip), axis=1)

        x = self.decoders[-1](x)
        # print(x.shape)
        x = self.final_conv(x)
        return self.tanh(x)
    


class PatchGAN(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.d1 = DownSampleConv(input_channels, 64, batchnorm=False)
        self.d2 = DownSampleConv(64, 128)
        self.d3 = DownSampleConv(128, 256)
        self.d4 = DownSampleConv(256, 512)
        self.final = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.d1(x)
        x1 = self.d2(x0)
        x2 = self.d3(x1)
        x3 = self.d4(x2)
        xn = self.final(x3)
        return xn
    

adversarial_loss = nn.BCEWithLogitsLoss()

reconstruction_loss = nn.L1Loss()




def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def display_progress(cond, real, fake, figsize=(10,5)):
    cond = cond.detach().cpu().permute(1, 2, 0)
    fake = fake.detach().cpu().permute(1, 2, 0)
    real = real.detach().cpu().permute(1, 2, 0)

    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].imshow(cond)
    ax[1].imshow(real)
    ax[2].imshow(fake)
    plt.show()



# Pix2Pix, CycleGAN, AttentionGAN (https://github.com/Ha0Tang/AttentionGAN?tab=readme-ov-file)




class Pix2Pix(pl.LightningModule):

    def __init__(self, in_channels, out_channels, learning_rate=0.0002, lambda_recon=200, display_step=25):

        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False

        self.display_step = display_step
        self.gen = Generator(in_channels, out_channels)
        self.patch_gan = PatchGAN(in_channels + out_channels)

        # intializing weights
        self.gen = self.gen.apply(_weights_init)
        self.patch_gan = self.patch_gan.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step(self, real_images, conditioned_images):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        fake_images = self.gen(conditioned_images)
        disc_logits = self.patch_gan(fake_images, conditioned_images)
        adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

        # calculate reconstruction loss
        recon_loss = self.recon_criterion(fake_images, real_images)
        lambda_recon = self.hparams.lambda_recon

        return adversarial_loss + lambda_recon * recon_loss

    def _disc_step(self, real_images, conditioned_images):
        fake_images = self.gen(conditioned_images).detach()
        fake_logits = self.patch_gan(fake_images, conditioned_images)

        real_logits = self.patch_gan(real_images, conditioned_images)

        fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.adversarial_criterion(real_logits, torch.ones_like(real_logits))
        return (real_loss + fake_loss) / 2

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        disc_opt = torch.optim.Adam(self.patch_gan.parameters(), lr=0.564861321564864)
        return disc_opt, gen_opt


    def validation_step(self, batch, batch_idx):

            # sketch_img, photo_img = batch
            real, condition = batch
            # outputs = self.G_basestyle(sketch_img)
            fake = self.gen(condition).detach()
            # ssim = self.default_evaluator.run([[fake, real]]).metrics['ssim']
            ssim = structural_similarity_index_measure(fake, real)
            psnr = peak_signal_noise_ratio(fake, real)
            self.log("SSIM_val", ssim)
            print(f"SSIM_val: {ssim}")
            self.log("PSNR_val", psnr)
            print(f"PSNR_val: {psnr}")
            # self.log("SSIM_valid", ssim)
            # return ssim

    def test_step(self, batch, batch_idx):
            
        # sketch_img, photo_img = batch
        real, condition = batch
        # outputs = self.G_basestyle(sketch_img)
        fake = self.gen(condition).detach()
        # ssim = self.default_evaluator.run([[fake, real]]).metrics['ssim']
        ssim = structural_similarity_index_measure(fake, real)
        psnr = peak_signal_noise_ratio(fake, real)
        # self.log("SSIM_train", ssim)
        self.log("SSIM_heldout_set", ssim)
        print(f"SSIM_heldout_set: {ssim}")
        self.log("PSNR_val", psnr)
        print(f"PSNR_val: {psnr}")
        # return ssim


    def training_step(self, batch, batch_idx):
        real, condition = batch

        disc_opt, gen_opt = self.optimizers()

        disc_loss = 0
        gen_loss = 0
        
        if batch_idx % 2 == 0:  # Train discriminator on even batches
            gen_opt.zero_grad()
            gen_loss = self._gen_step(real, condition)
            # loss = self._disc_step(real, condition)
            self.manual_backward(gen_loss)
            gen_opt.step()
            print('Generator Loss', gen_loss)
            self.log("gen_train_loss", gen_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
        else:
            disc_opt.zero_grad()
            # loss = self._gen_step(real, condition)
            disc_loss = self._disc_step(real, condition)
            self.manual_backward(disc_loss)
            disc_opt.step()
            print('PatchGAN Loss', disc_loss)
            self.log("disc_train_loss", disc_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)




        if self.current_epoch%self.display_step==0 and batch_idx==0:
            # print('PatchGAN Loss', disc_loss)
            # print('Generator Loss', gen_loss)
            fake = self.gen(condition).detach()

            import random

            # Generate 5 random numbers between 0 and 49 (inclusive)
            random_numbers = random.sample(range(len(condition)-2), 5)

            for i in random_numbers:
                display_progress(condition[i], fake[i], real[i])
            # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            print('-'*100)

        # return loss