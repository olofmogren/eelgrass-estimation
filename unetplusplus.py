import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Standard VGG-style convolutional block: (Conv -> BN -> ReLU) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

class Model(nn.Module):
    """
    U-Net++ model architecture with support for an auxiliary feature output for
    calculating an invariance loss during training.
    """
    def __init__(self, in_channels=3, out_channels=1, nb_filter=(32, 64, 128, 256, 512)):
        super(Model, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Level 0
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        # Level 1
        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        # Level 2
        self.conv0_2 = ConvBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2]*2 + nb_filter[3], nb_filter[2], nb_filter[2])

        # Level 3
        self.conv0_3 = ConvBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1]*3 + nb_filter[2], nb_filter[1], nb_filter[1])

        # Level 4
        self.conv0_4 = ConvBlock(nb_filter[0]*4 + nb_filter[1], nb_filter[0], nb_filter[0])

        # --- THIS IS THE FIX ---
        self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, input):
        # -- Encoder Path --
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        # -- Bottleneck --
        x4_0 = self.conv4_0(self.pool(x3_0))

        # -- Decoder Path --
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # Conditionally return the feature map based on the model's mode.
        # Your training loop expects two outputs (logits, features) during training
        # and one output (logits) during evaluation.
        #if self.deep_supervision:
        logits1 = self.final1(x0_1)
        logits2 = self.final2(x0_2)
        logits3 = self.final3(x0_3)
        logits4 = self.final4(x0_4)
        return logits1, logits2, logits3, logits4, x0_4

