"""
Discriminator and Generator for Wasserstein GAN
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_channels, features_dim):
        super(Discriminator, self).__init__()
        #input size = 64 * 64
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, features_dim, kernel_size=4, stride=2, padding=1), # 32 X 32
            nn.LeakyReLU(0.2),
            self._block(features_dim, features_dim*2, 4, 2, 1),     #16X16
            self._block(features_dim*2, features_dim*4, 4, 2, 1),    #8X8
            self._block(features_dim*4, features_dim*8, 4, 2, 1),   #4X4
            nn.Conv2d(features_dim*8, 1, 4, 2, padding=0),          #1X1

        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size,
            stride,
            padding,
            bias = False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, img_channels, noise_channels, features_gen):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(noise_channels, features_gen*16, 4, 1, 0),       #4X4
            self._block(features_gen*16, features_gen*8, 4, 2, 1),       #8X8
            self._block(features_gen*8, features_gen*4, 4, 2, 1),        #16*16
            self._block(features_gen*4, features_gen*2, 4, 2, 1),        #32X32
            nn.ConvTranspose2d(features_gen*2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64      # batchsize, in_channels, height, width
    noise_dim = 100                          # arbitrary noise dimension value
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Incorrect Discriminator output shape"
    gen = Generator(in_channels, noise_dim, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Incorrect Generator output shape"
    print("tests passed")


if __name__ == "__main__":
    print("Running tests")
    test()












    