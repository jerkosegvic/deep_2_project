import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.module(x)


class TimeMLP(nn.Module):
    def __init__(self, embedding_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, out_dim)
        )

    def forward(self, x, t):
        x = x + self.mlp(t).unsqueeze(-1).unsqueeze(-1)
        return torch.relu(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBlock(
            in_channels // 2, out_channels // 2, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.match_dim = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   padding=0) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        skip_connection = self.match_dim(x)
        x = self.conv_block(x)
        x = self.bn2(self.conv2(x))

        return self.activation(skip_connection + x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, padding=1)

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            *[ResidualBlock(in_channels, in_channels) for i in range(3)],
            ResidualBlock(in_channels, out_channels // 2)
        )
        self.time_mlp = TimeMLP(
            embedding_dim=embedding_dim, out_dim=out_channels // 2)
        self.conv2 = DownsampleBlock(out_channels // 2, out_channels)

    def forward(self, x, t=None):
        x_skip = self.conv1(x)
        if t is not None:
            x = self.time_mlp(x_skip, t)
        x = self.conv2(x)

        return x, x_skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Sequential(
            *[ResidualBlock(in_channels, in_channels) for i in range(3)],
            ResidualBlock(in_channels, in_channels // 2)
        )
        self.time_mlp = TimeMLP(
            embedding_dim=embedding_dim, out_dim=in_channels // 2)
        self.conv2 = DownsampleBlock(in_channels // 2, out_channels)

    def forward(self, x, x_skip, t=None):
        x = self.upsample(x)
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv1(x)
        if t is not None:
            x = self.time_mlp(x, t)
        x = self.conv2(x)

        return x


class UNet(nn.Module):
    def __init__(self, timesteps, embedding_dim, in_channels=3, out_channels=2, base_dim=32, dim_mults=[2, 4, 8, 16]):
        super().__init__()

        self.conv1 = ConvBlock(3, base_dim, padding=1)
        self.time_embedding = nn.Embedding(timesteps, embedding_dim)

        dims = [base_dim, *[base_dim * mult for mult in dim_mults]]
        channels = [(x, y) for x, y in zip(dims[:-1], dims[1:])]

        self.encoder = nn.ModuleList(
            [EncoderBlock(c_in, c_out, embedding_dim) for c_in, c_out in channels])
        self.decoder = nn.ModuleList(
            [DecoderBlock(c_out, c_in, embedding_dim)
             for c_in, c_out in channels[::-1]]
        )

        out_dim = channels[-1][1]
        self.mid_block = nn.Sequential(*[ResidualBlock(out_dim, out_dim) for i in range(2)],
                                       ResidualBlock(out_dim, out_dim // 2))

        self.conv2 = nn.Conv2d(
            channels[0][0] // 2, out_channels, kernel_size=1)

    def forward(self, x, t):
        x = self.conv1(x)
        if t is not None:
            t = self.time_embedding(t)

        encoder_skips = []
        for encoder_block in self.encoder:
            x, x_skip = encoder_block(x, t)
            encoder_skips.append(x_skip)

        x = self.mid_block(x)

        encoder_skips.reverse()
        for decoder_block, skip in zip(self.decoder, encoder_skips):
            # print(skip.shape, x.shape)
            x = decoder_block(x, skip, t)

        x = self.conv2(x)
        return x


if __name__ == "__main__":
    x = torch.randn(1, 3, 128, 128)
    t = torch.randint(0, 1000, (1,))
    model = UNet(1000, 128)
    y = model(x, t)
    print(y.shape)
