import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=0, 
            activation=nn.ReLU,
            groups=1
        ):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels,
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding,
                groups=groups
            ),
            nn.BatchNorm2d(out_channels),
            activation()
        )

    def forward(self, x):
        return self.module(x)

class ShuffleBlock(nn.Module):
    def __init__(self,groups):
        super().__init__()
        self.groups=groups

    def forward(self,x):
        N,C,H,W=x.size()
        g=self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)
    
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


class MainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.SiLU):
        super().__init__()
        self.module = nn.Sequential(
            ConvBlock(
                in_channels, 
                in_channels, 
                padding=0,
                kernel_size=1,
                activation=activation,
            ),
            ConvBlock(
                in_channels,
                in_channels,
                padding="same",
                activation=torch.nn.Identity,
                groups=in_channels
            ),
            ConvBlock(
                in_channels,
                out_channels,
                padding=0,
                kernel_size=1,
                activation=activation
            ),
            #ShuffleBlock(groups=2)
        )            

    def forward(self, x):
        return self.module(x)

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.SiLU):
        super().__init__()
        self.module = nn.Sequential(
            ConvBlock(
                in_channels, 
                out_channels, 
                padding=0,
                kernel_size=1,
                activation=activation,
            ),
            ConvBlock(
                out_channels,
                out_channels,
                padding=1,
                stride=2,
                activation=activation,
                groups=out_channels
            ),
            ConvBlock(
                out_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                activation=activation
            ),
            #ShuffleBlock(groups=2)
        )

    def forward(self, x):
        return self.module(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            *[MainBlock(in_channels, in_channels) for i in range(3)],
            MainBlock(in_channels, out_channels // 2)
        )
        self.time_mlp = TimeMLP(
            embedding_dim=embedding_dim, 
            out_dim=out_channels // 2
        )
        
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
            scale_factor=2, 
            mode='bilinear', 
            align_corners=False
        )
        self.conv1 = nn.Sequential(
            *[MainBlock(in_channels, in_channels) for i in range(3)],
            MainBlock(in_channels, in_channels // 2)
        )
        self.time_mlp = TimeMLP(
            embedding_dim=embedding_dim, 
            out_dim=in_channels // 2
        )
        self.conv2 = MainBlock(in_channels // 2, out_channels // 2)

    def forward(self, x, x_skip, t=None):
        x = self.upsample(x)
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv1(x)
        if t is not None:
            x = self.time_mlp(x, t)
        x = self.conv2(x)

        return x


class Unet(nn.Module):
    def __init__(
        self, 
        timesteps, 
        embedding_dim, 
        in_channels=3, 
        out_channels=2, 
        base_dim=32, 
        dim_mults=[2, 4, 8, 16]
    ):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, base_dim, kernel_size=3, padding="same")
        self.time_embedding = nn.Embedding(timesteps, embedding_dim)

        dims = [base_dim, *[base_dim * mult for mult in dim_mults]]
        channels = [(x, y) for x, y in zip(dims[:-1], dims[1:])]

        self.encoder = nn.ModuleList(
            [EncoderBlock(c_in, c_out, embedding_dim) for c_in, c_out in channels]
        )
        self.decoder = nn.ModuleList(
            [DecoderBlock(c_out, c_in, embedding_dim) for c_in, c_out in channels[::-1]] # reverse
        )

        out_dim = channels[-1][1]
        self.mid_block = nn.Sequential(*[MainBlock(out_dim, out_dim) for i in range(2)],
                                       MainBlock(out_dim, out_dim // 2))

        self.conv2 = nn.Conv2d(channels[0][0] // 2, out_channels, kernel_size=1)

    def forward(self, x, t):
        x = self.conv1(x)
        if t is not None:
            t = self.time_embedding(t)

        encoder_skips = []
        for encoder_block in self.encoder:
            x, x_skip = encoder_block(x, t)
            encoder_skips.append(x_skip)

        x = self.mid_block(x)
        #breakpoint()
        encoder_skips.reverse()
        for decoder_block, skip in zip(self.decoder, encoder_skips):
            x = decoder_block(x, skip, t)

        x = self.conv2(x)
        return x


if __name__ == "__main__":
    x = torch.randn(1, 3, 128, 128)
    t = torch.randint(0, 1000, (1,))
    model = Unet(1000, 128)
    y = model(x, t)
    print(y.shape)
