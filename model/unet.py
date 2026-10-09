import torch
from model.utils import modulate

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels=0, device='cpu', ada=False, dropout=0.0, init_first=True):
        super().__init__()
        self.use_ada = ada

        if self.use_ada:
            if init_first:
                self.norm1 = torch.nn.LayerNorm(in_channels, elementwise_affine=False, device=device)
            else:
                self.norm1 = None
            self.norm2 = torch.nn.LayerNorm(out_channels, elementwise_affine=False, device=device)
            self.film_params_1 = torch.nn.Sequential(
                torch.nn.GELU(),
                torch.nn.Linear(emb_channels, 2 * in_channels, bias=True, device=device)
            )
            self.film_params_2 = torch.nn.Sequential(
                torch.nn.GELU(),
                torch.nn.Linear(emb_channels, 2 * out_channels, bias=True, device=device)
            )
        else:
            self.norm1 = torch.nn.LayerNorm(in_channels, device=device)
            self.norm2 = torch.nn.LayerNorm(out_channels, device=device)
        self.act = torch.nn.GELU()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), device=device)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), device=device)
        self.dropout2d = torch.nn.Dropout2d(dropout)

        self.skip = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1),
                                      device=device) if in_channels != out_channels else torch.nn.Identity()

    def forward(self, x, emb_t=None):
        identity = self.skip(x)

        if not self.use_ada:
            x_in = self.norm1(x.transpose(1, -1)).transpose(1, -1)
            x = self.conv1(self.act(x_in))
            x_in = self.norm2(x.transpose(1, -1)).transpose(1, -1)
            x = self.conv2(self.act(x_in))
        else:
            shift_1, scale_1 = self.film_params_1(emb_t.transpose(1, 2).transpose(2, 3)).chunk(2, dim=-1)
            shift_2, scale_2 = self.film_params_2(emb_t.transpose(1, 2).transpose(2, 3)).chunk(2, dim=-1)
            if self.norm1 is not None:
                x_in = modulate(self.norm1(x.transpose(1, -1)).transpose(1, -1),
                                shift_1.transpose(1, -1).repeat(1, 1, x.shape[-2], x.shape[-1]).transpose(1, 2).transpose(2, 3),
                                scale_1.transpose(1, -1).repeat(1, 1, x.shape[-2], x.shape[-1]).transpose(1, 2).transpose(2, 3))
            else:
                x_in = x
            x = self.conv1(self.act(x_in))
            x_in = modulate(self.norm2(x.transpose(1, -1)).transpose(1, -1),
                            shift_2.transpose(1, -1).repeat(1, 1, x.shape[-2], x.shape[-1]).transpose(1, 2).transpose(2, 3),
                            scale_2.transpose(1, -1).repeat(1, 1, x.shape[-2], x.shape[-1]).transpose(1, 2).transpose(2, 3))
            x = self.conv2(self.act(x_in))

        return self.dropout2d(x) + identity


class Down(torch.nn.Module):
    def __init__(self, in_channels, out_channels, device='cpu', ada=False, dropout=0.0, emb_channels=0, 
                 init_first=True):
        super().__init__()
        self.block = ResidualBlock(in_channels, out_channels, device=device, ada=ada, dropout=dropout, 
                                   emb_channels=emb_channels, init_first=init_first)
        self.pool = torch.nn.MaxPool2d((1, 2))

    def forward(self, x, emb_t=None):
        x = self.block(x, emb_t=emb_t)
        x_down = self.pool(x)
        return x, x_down


class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels, device='cpu', ada=False, dropout=0.0, emb_channels=0):
        super().__init__()
        self.upsample = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(1, 2), stride=(1, 2),
                                                   device=device)
        self.block = ResidualBlock(in_channels * 2, out_channels, device=device, ada=ada, dropout=dropout,
                                   emb_channels=emb_channels)

    def forward(self, x, skip, emb_t=None):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x, emb_t=emb_t)


class UNet(torch.nn.Module):
    def __init__(self, in_channels=3, features=[32,32], device='cpu', ada=False, dropout=0.0, emb_channels=0):
        super().__init__()
        self.encoder_blocks = torch.nn.ModuleList()
        self.decoder_blocks = torch.nn.ModuleList()

        # Encoder
        channels = in_channels
        self.downs = torch.nn.ModuleList()
        for i, feature in enumerate(features):
            if i == 0:
                down_block = Down(channels, feature, device=device, ada=ada, dropout=dropout, emb_channels=emb_channels,
                                  init_first=False)
            else:
                down_block = Down(channels, feature, device=device, ada=ada, dropout=dropout, emb_channels=emb_channels)
            self.downs.append(down_block)
            channels = feature

        # Bottleneck
        self.bottleneck = ResidualBlock(features[-1], features[-1], device=device, ada=ada, dropout=dropout, 
                                        emb_channels=emb_channels)

        # Decoder
        reversed_features = list(reversed(features))
        channels = features[-1]
        for feature in reversed_features:
            up_block = Up(channels, feature, device=device, ada=ada, dropout=dropout, emb_channels=emb_channels)
            self.decoder_blocks.append(up_block)
            channels = feature

    def forward(self, x, emb_t=None):
        skips = []

        # Encoder path
        for down in self.downs:
            x, x_down = down(x, emb_t=emb_t)
            skips.append(x)
            x = x_down

        x = self.bottleneck(x, emb_t=emb_t)

        # Decoder path
        for up, skip in zip(self.decoder_blocks, reversed(skips)):
            x = up(x, skip, emb_t=emb_t)

        return x 
